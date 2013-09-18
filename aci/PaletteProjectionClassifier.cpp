#include "PaletteProjectionClassifier.h"

PaletteProjectionClassifier::PaletteProjectionClassifier(int nbBins, double magnification, double suppression)
	: nbBins(nbBins), magnification(magnification), suppression(suppression), internalClassifier(true)
{
	lrange[0] = 0;
	lrange[1] = 100;
	arange[0] = -127;
	arange[1] = 127;
	brange[0] = -127;
	brange[1] = 127;
}

static bool compHistEntry(const PaletteProjectionClassifier::HistEntry &h1, const PaletteProjectionClassifier::HistEntry &h2) {
	return get<3>(h1) < get<3>(h2);
}

void PaletteProjectionClassifier::computePalette(const Mat_<Vec3f> &image, const Mat_<float> &mask, VectorXd &palette, vector<HistEntry> &flatHistogram) {
	// compute L*a*b* color histogram
	// quantize each channel with nbBins
	int lbins = this->nbBins, abins = this->nbBins, bbins = this->nbBins;
	int histSize[] = {lbins, abins, bbins};
	const float *ranges[] = {lrange, arange, brange};
	MatND hist;

	int channels[] = {0, 1, 2};

	calcHist(&image, 1, channels, Mat_<uchar>(mask*255), hist, 3, histSize, ranges, true, false);

	// sort histogram
	flatHistogram.clear();
	flatHistogram.reserve(lbins * abins * bbins);
	int index = 0;

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

int PaletteProjectionClassifier::get1DBin(float x, float range[]) {
	return (int)(((x - range[0]) / (range[1] - range[0])) * (this->nbBins - 1));
}

void PaletteProjectionClassifier::paletteRewrite(const Mat_<Vec3f> &image, const Mat_<float> &mask, const VectorXd &newPalette, const vector<HistEntry> &sortedHistogram, Mat_<Vec3f> &repaletted) {
	repaletted = Mat_<Vec3f>::zeros(image.rows, image.cols);
	// associate to each bin the corresponding new color
	vector<vector<vector<Vec3f> > > histogram;
	histogram.reserve(this->nbBins);

	for (int i = 0; i < this->nbBins; i++) {
		histogram.push_back(vector<vector<Vec3f> >());
		histogram[i].reserve(this->nbBins);

		for (int j = 0; j < this->nbBins; j++) {
			histogram[i].push_back(vector<Vec3f>(this->nbBins, 0));
		}
	}

	for (int i = 0; i < (int)sortedHistogram.size(); i++) {
		int lbin, abin, bbin;
		std::tie(lbin, abin, bbin, std::ignore) = sortedHistogram[i];

		histogram[lbin][abin][bbin] = Vec3f(newPalette(3 * i), newPalette(3 * i + 1), newPalette(3 * i + 2));
	}

	// for each pixel in the source image, identify the bin and add the corresponding
	// color.
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (mask(i,j) > 0) {
				Vec3f pixColor = image(i,j);

				int lbin = this->get1DBin(pixColor[0], lrange);
				int abin = this->get1DBin(pixColor[1], arange);
				int bbin = this->get1DBin(pixColor[2], brange);

				repaletted(i,j) = histogram[lbin][abin][bbin];
			}
		}
	}
}

void PaletteProjectionClassifier::train(vector<TrainingSample> &samples) {
	// compute palettes
	MatrixXd palettes(3 * this->nbBins * this->nbBins * this->nbBins, samples.size());
	vector<vector<HistEntry> > flatHistograms(samples.size());

	for (int i = 0; i < (int)samples.size(); i++) {
		VectorXd palette;
		
		this->computePalette(get<1>(samples[i]), get<2>(samples[i]), palette, flatHistograms[i]);

		palettes.col(i) = palette;
	}

	// determine sigma by the variance of euclid distances between all pairs
	MatrixXd euclidDistances(samples.size(), samples.size());
	double mean = 0;
	double sigma2 = 0;

	for (int i = 0; i < (int)samples.size(); i++) {
		for (int j = i + 1; j < (int)samples.size(); j++) {
			euclidDistances(i,j) = (palettes.col(i) - palettes.col(j)).norm();

			mean += euclidDistances(i,j);
			sigma2 += pow(euclidDistances(i,j), 2);
		}
	}

	int nbPairs = (samples.size() * (samples.size() + 1)) / 2;

	mean = mean / nbPairs;
	sigma2 = (sigma2/nbPairs) - pow(mean, 2);

	// compute modulated similarity matrix
	MatrixXd similarity = MatrixXd::Zero(samples.size(), samples.size());

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
	GeneralizedSelfAdjointEigenSolver<MatrixXd> solver(palettes * similarity * palettes.transpose(), PFP, ComputeEigenvectors);

	this->projection = solver.eigenvectors().transpose();

	// compute projection palettes for each sample, and rewrite the image with the new
	// palette into the training set for the matching segment classifier.
	vector<TrainingSample> repalettedSamples;
	repalettedSamples.reserve(samples.size());

	for (int i = 0; i < samples.size(); i++) {
		VectorXd projectionPalette = this->projection * palettes.col(i);
		Mat_<Vec3f> repaletted;

		this->paletteRewrite(get<1>(samples[i]), get<2>(samples[i]), projectionPalette, flatHistograms[i], repaletted);

		repalettedSamples.push_back(TrainingSample(get<0>(samples[i]), repaletted, get<2>(samples[i]), get<3>(samples[i])));
	}

	this->internalClassifier.train(repalettedSamples);
}

int PaletteProjectionClassifier::predict(DisjointSetForest &segmentation, const Mat_<Vec3f> &image, const Mat_<float> &mask) {
	// compute palette
	VectorXd palette;
	vector<HistEntry> flatHistogram;

	this->computePalette(image, mask, palette, flatHistogram);
	// project palette
	VectorXd projectedPalette = this->projection * palette;
	// apply new palette
	Mat_<Vec3f> repaletted;

	this->paletteRewrite(image, mask, projectedPalette, flatHistogram, repaletted);

	return this->internalClassifier.predict(segmentation, repaletted, mask);
}
