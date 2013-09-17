#include "PaletteProjectionClassifier.h"

PaletteProjectionClassifier::PaletteProjectionClassifier(int nbBins, double magnification, double suppression, double sigma)
	: nbBins(nbBins), magnification(magnification), suppression(suppression), internalClassifier(true), sigma(sigma)
{
}

typedef std::tuple<int,int,int,int> HistEntry;

static bool compHistEntry(const HistEntry &h1, const HistEntry &h2) {
	return get<3>(h1) < get<3>(h2);
}

void PaletteProjectionClassifier::computePalette(const Mat_<Vec3f> &image, const Mat_<float> &mask, VectorXd &palette) {
	// compute L*a*b* color histogram
	// quantize each channel with nbBins
	int lbins = this->nbBins, abins = this->nbBins, bbins = this->nbBins;
	int histSize[] = {lbins, abins, bbins};
	float lrange[] = {0, 100};
	float arange[] = {-127, 127};
	float brange[] = {-127, 127};
	const float *ranges[] = {lrange, arange, brange};
	MatND hist;

	int channels[] = {0, 1, 2};

	calcHist(&image, 1, channels, mask, hist, 3, histSize, ranges, true, false);

	// sort histogram
	vector<HistEntry> flatHistogram;
	flatHistogram.reserve(lbins * abins * bbins);

	for (int i = 0; i < lbins; i++) {
		for (int j = 0; j < abins; j++) {
			for (int k = 0; k < bbins; k++) {
				flatHistogram.push_back(HistEntry(i,j,k,hist.at<int,3>(Vec<int,3>(i,j,k))));
			}
		}
	}

	sort(flatHistogram.begin(), flatHistogram.end(), compHistEntry);

	// build palette using sorted histogram with median color (not average)
	palette = VectorXd::Zero(3 * lbins * abins * bbins);

	for (int i = 0; i < lbins * abins * bbins; i++) {
		int lbin, abin, bbin;
		std::tie(lbin, abin, bbin, std::ignore) = flatHistogram[i];
		float lratio = (float)lbin/(float)lbins;
		float aratio = (float)abin/(float)abins;
		float bratio = (float)bbin/(float)bbins;
		palette(3 * i) = lrange[0] + (lrange[1] - lrange[0]) * lratio;
		palette(3 * i + 1) = arange[0] + (arange[1] - arange[0]) * aratio;
		palette(3 * i + 2) = brange[0] + (brange[1] - brange[0]) * bratio;
	}
}

void PaletteProjectionClassifier::train(vector<TrainingSample> &samples) {
	// compute palettes
	MatrixXd palettes(this->nbBins, samples.size());

	for (int i = 0; i < (int)samples.size(); i++) {
		VectorXd palette;

		this->computePalette(get<1>(samples[i]), get<2>(samples[i]), palette);

		palettes.col(i) = palette;
	}

	// compute modulated similarity matrix
	MatrixXd similarity = MatrixXd::Zero(samples.size(), samples.size());
	double sigma2 = pow(sigma, 2);

	for (int i = 0; i < (int)samples.size(); i++) {
		for (int j = i + 1; j < (int)samples.size(); j++) {
			similarity(i,j) = exp(-(palettes.col(i) - palettes.col(j)).squaredNorm() / sigma2);
			if (get<3>(samples[i]) >= 0 && get<3>(samples[j]) >= 0) {
				if (get<3>(samples[i]) == get<3>(samples[j])) {
					similarity(i,j) = this->magnification * similarity(i,j);
				} else {
					similarity(i,j) = this->suppression * similarity(i,j);
				}
			}
			similarity(j,i) = similarity(i,j);
		}
	}

	// compute degree matrix
	MatrixXd degrees = MatrixXd::Zero(samples.size(), samples.size());

	for (int i = 0; i < (int)samples.size(); i++) {
		degrees(i,i) = similarity.row(i).sum();
	}

	// regularize PFP matrix just in case
	MatrixXd PFP = palettes * degrees * palettes.transpose() + 10E-14 * MatrixXd::Identity(samples.size(), samples.size());

	// solve generalized eigenvalue problem
	GeneralizedSelfAdjointEigenSolver<MatrixXd> solver(palettes * similarity * palettes.transpose(), PFP);

	this->projection = solver.eigenvectors().transpose();
}
