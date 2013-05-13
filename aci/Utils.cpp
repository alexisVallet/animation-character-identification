#include "Utils.hpp"

int toRowMajor(int width, int x, int y) {
  return x + width * y;
}

int toColumnMajor(int rows, int i, int j) {
	return i + j * rows;
}

pair<int,int> fromRowMajor(int width, int i) {
  pair<int,int> coords(i/width, i%width);

  return coords;
}

static bool isMask(const Mat_<float> &mask) {
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (mask(i,j) != 0 && mask(i,j) != 1) {
				cout<<"failed: mask("<<i<<","<<j<<") = "<<mask(i,j)<<endl;

				return false;
			}
		}
	}

	return true;
}

void loadDataSet(char* folderName, char** charaNames, int nbCharas, int nbImagesPerChara, vector<pair<Mat_<Vec<uchar,3> >,Mat_<float> > > &images, Mat_<int> &classes) {
	images = vector<pair<Mat_<Vec<uchar,3> >,Mat_<float> > >(nbCharas * nbImagesPerChara);
	classes = Mat_<int>(nbCharas * nbImagesPerChara, 1);

	for (int i = 0; i < nbCharas; i++) {		
		for (int j = 0; j < nbImagesPerChara; j++) {
			char suffix[] = {'_', 'a' + j, '.', 'p', 'n', 'g', '\0'};
			char *fullPath = new char[strlen(folderName) + strlen(charaNames[i]) + strlen(suffix) + 1];
			char maskSuffix[] = {'-', 'm', 'a', 's', 'k', '.', 'p', 'n', 'g', '\0'};
			char *maskPath = new char[strlen(folderName) + strlen(charaNames[i]) + strlen(suffix) + strlen(maskSuffix) + 1];
			
			strcpy(fullPath, folderName);
			strcat(fullPath, charaNames[i]);
			strcat(fullPath, suffix);
			strcpy(maskPath, fullPath);
			strcat(maskPath, maskSuffix);

			int rowMajorIndex = toRowMajor(nbImagesPerChara, j, i);

			Mat_<Vec<uchar, 3> > mask = imread(maskPath);
			vector<Mat_<uchar> > maskChannels;

			split(mask, maskChannels);

			images[rowMajorIndex].first = imread(fullPath);

			Mat_<uchar> thresholdedMask;

			threshold(maskChannels[0], thresholdedMask, 128, 1, THRESH_BINARY_INV);

			images[rowMajorIndex].second = Mat_<float>(thresholdedMask);

			crop(images[rowMajorIndex].first, images[rowMajorIndex].second, images[rowMajorIndex].first, images[rowMajorIndex].second);

			assert(isMask(images[rowMajorIndex].second));

			classes(rowMajorIndex,0) = i;

			delete[] fullPath;
			delete[] maskPath;
		}
	}
}

Mat_<double> sparseMul(SparseMat_<double> A, Mat_<double> b) {
	assert(A.size(1) == b.rows);
	assert(b.cols == 1);
	Mat_<double> c = Mat_<double>::zeros(b.rows, 1);

	SparseMatConstIterator_<double> it;

	// iterates over non zero elements
	for (it = A.begin(); it != A.end(); ++it) {
		const SparseMat_<double>::Node* n = it.node();
		int row = n->idx[0];
		int col = n->idx[1];

		c(row, 0) += it.value<double>() * b(col,0);
	}

	return c;
}

bool symmetric(Eigen::SparseMatrix<double> M) {
	for (int k = 0; k < M.outerSize(); k++) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
			if (abs(it.value() - M.coeffRef(it.col(), it.row())) > 0) {
				return false;
			}
		}
	}

	return true;
}

bool positiveDefinite(Eigen::SparseMatrix<double> M) {
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > chol;

	chol.compute(M);

	return chol.info() == Eigen::Success;
}

// Remove a line and column with the same index in a sparse matrix.
void removeLineCol(const Eigen::SparseMatrix<double> &L, int v0, Eigen::SparseMatrix<double> &L0) {
	typedef Eigen::Triplet<double> T;
	vector<T> tripletList;
	tripletList.reserve(L.nonZeros());

	for (int k = 0; k < L.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(L,k); it; ++it) {
			if (it.row() != v0 && it.col() != v0) {
				int newRow = it.row() < v0 ? it.row() : it.row() - 1;
				int newCol = it.col() < v0 ? it.col() : it.col() - 1;

				tripletList.push_back(T(newRow, newCol, it.value()));
			}
		}
	}

	L0 = Eigen::SparseMatrix<double>(L.rows() - 1, L.cols() - 1);

	L0.setFromTriplets(tripletList.begin(), tripletList.end());
}

int toUpperTriangularPacked(int i, int j) {
	if (i > j) {
		return toUpperTriangularPacked(j, i);
	} else {
		int result = i + (j + 1) * j / 2;

		return result;
	}
}

Mat imHist(Mat hist, float scaleX, float scaleY){
	double maxVal=0;
	minMaxLoc(hist, 0, &maxVal, 0, 0);
	int rows = 64; //default height size
	int cols = hist.rows; //get the width size from the histogram
	Mat histImg = Mat::zeros((uchar)(rows*scaleX), (uchar)(cols*scaleY), CV_8UC3);
	//for each bin
	for(int i=0;i<cols-1;i++) {
		float histValue = hist.at<float>(i,0);
		float nextValue = hist.at<float>(i+1,0);
		Point pt1 = Point((int)(i*scaleX), (int)(rows*scaleY));
		Point pt2 = Point((int)(i*scaleX+scaleX), (int)(rows*scaleY));
		Point pt3 = Point((int)(i*scaleX+scaleX), (int)((rows-nextValue*rows/maxVal)*scaleY));
		Point pt4 = Point((int)(i*scaleX), (int)((rows-nextValue*rows/maxVal)*scaleY));

		int numPts = 5;
		Point pts[] = {pt1, pt2, pt3, pt4, pt1};

		fillConvexPoly(histImg, pts, numPts, Scalar(255,255,255));
	}
	return histImg;
}

void showHistograms(const Mat_<Vec3b> &image, const Mat_<float> &mask, int nbBins) {
	vector<Mat> channels;
	Mat newMask = Mat_<uchar>(mask);

	split(image, channels);

	for (int i = 0; i < (int)channels.size(); i++) {
		Mat histogram;
		int channelInd[1] = {0};
		int histSize[] = {nbBins};
		float hrange[] = {0, 256};
		const float *ranges[] = {hrange};

		calcHist(&channels[i], 1, channelInd, newMask, histogram, 1, histSize, ranges);
		Mat histogramDrawing = imHist(histogram);

		stringstream ss;

		ss<<"channel "<<i;

		imshow(ss.str(), histogramDrawing);
	}
}

static void equalizeGrayscaleHistogram(const Mat_<uchar> &image, const Mat_<float> &mask, Mat_<uchar> &equalized) {
	// first compute the histogram of the non masked elements
	Mat_<uchar> ucharMask = Mat_<uchar>(mask);

	int bins = 256;
	int histSize[] = {bins};
	float range[] = {0, 256};
	const float* ranges[] = {range};
	Mat_<float> histogram;
	int channels[] = {0};

	calcHist(&image, 1, channels, ucharMask, histogram, 1, histSize, ranges);

	// normalize the histogram
	Mat_<float> normalized;

	normalize(histogram, normalized, 255, 0, NORM_L1);

	// compute the accumulated normalized histogram
	Mat_<float> accumulated = Mat_<float>::zeros(histogram.rows, 1);
	accumulated(0,0) = normalized(0,0);

	for (int i = 1; i < histogram.rows; i++) {
		accumulated(i,0) = accumulated(i-1,0) + normalized(i,0);
	}
	// compute the equalized image from the accumulated histogram
	equalized = Mat_<uchar>(image.rows, image.cols);

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			int index = image(i,j);

			equalized(i,j) = (uchar)accumulated(index);
		}
	}
}

void equalizeColorHistogram(const Mat_<Vec3b> &image, const Mat_<float> &mask, Mat_<Vec3b> &equalized) {
	Mat_<Vec3b> hsvImage;

	cout<<"converting to HSV"<<endl;
	cvtColor(image, hsvImage, CV_BGR2HSV);

	cout<<"splitting"<<endl;
	vector<Mat_<uchar> > channels(3);

	split(image, channels);

	cout<<"equalizing Hue"<<endl;
	Mat_<uchar> equalizedHue;

	equalizeGrayscaleHistogram(channels[0], mask, equalizedHue);

	channels[0] = equalizedHue;

	cout<<"merging back equalized hsv"<<endl;
	Mat_<Vec3b> equalizedHsv;
	
	for (int i = 1; i < 3; i++) {
		channels[i] = channels[i].mul(Mat_<uchar>(mask));
	}

	merge(channels, equalizedHsv);

	cout<<"converting back to BGR"<<endl;
	cvtColor(equalizedHsv, equalized, CV_HSV2BGR);
}

void crop(const Mat_<Vec3b> &image, const Mat_<float> &mask, Mat_<Vec3b> &croppedImage, Mat_<float> &croppedMask) {
	assert(image.rows == mask.rows && image.cols == mask.cols);
	assert(countNonZero(mask) > 0);
	int minI = image.rows;
	int maxI = -1;
	int minJ = image.cols;
	int maxJ = -1;

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (mask(i,j) > 0) {
				minI = min(i, minI);
				maxI = max(i, maxI);
				minJ = min(j, minJ);
				maxJ = max(j, maxJ);
			}
		}
	}

	croppedImage = image.rowRange(minI, maxI + 1).colRange(minJ, maxJ + 1);
	croppedMask = mask.rowRange(minI, maxI + 1).colRange(minJ, maxJ + 1);
}

void resizeImage(const Mat_<Vec<uchar,3> > &image, const Mat_<float> &mask, Mat_<Vec<uchar,3> > &resizedImage, Mat_<float> &resizedMask, int maxNbPixels) {
	assert(maxNbPixels >= 0);
	int nbPixels = countNonZero(mask);

	if (nbPixels > maxNbPixels) {
		double ratio = sqrt((double)maxNbPixels / (double)nbPixels);

		resize(image, resizedImage, Size(), ratio, ratio);
		resize(mask, resizedMask, Size(), ratio, ratio, INTER_NEAREST);
	} else {
		resizedImage = image;
		resizedMask = mask;
	}
}
