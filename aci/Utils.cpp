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

void loadDataSet(char* folderName, char** charaNames, int nbCharas, int nbImagesPerChara, vector<pair<Mat_<Vec<uchar,3> >,Mat_<float> > > &images, Mat &classes) {
	images = vector<pair<Mat_<Vec<uchar,3> >,Mat_<float> > >(nbCharas * nbImagesPerChara);
	classes = Mat(nbCharas * nbImagesPerChara, 1, CV_32S);

	for (int i = 0; i < nbCharas; i++) {		
		for (int j = 0; j < nbImagesPerChara; j++) {
            char letter = 'a' + j;
			char suffix[] = {'_', letter, '.', 'p', 'n', 'g', '\0'};
			char *fullPath = new char[strlen(folderName) + strlen(charaNames[i]) + strlen(suffix) + 1];
			char maskSuffix[] = {'_', letter,'-', 'm', 'a', 's', 'k', '.', 'p', 'n', 'g', '\0'};
			char *maskPath = new char[strlen(folderName) + strlen(charaNames[i]) + strlen(maskSuffix) + 1];
			
			strcpy(fullPath, folderName);
			strcat(fullPath, charaNames[i]);
            strcpy(maskPath, fullPath);
			strcat(fullPath, suffix);
			strcat(maskPath, maskSuffix);

			int rowMajorIndex = toRowMajor(nbImagesPerChara, j, i);

			Mat_<Vec<uchar, 3> > mask = imread(maskPath);
			vector<Mat_<uchar> > maskChannels;

			split(mask, maskChannels);

			images[rowMajorIndex].first = imread(fullPath);
			images[rowMajorIndex].second = (Mat_<float>(maskChannels[0]) / 255);
			classes.at<int>(rowMajorIndex,0) = i;

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
	Mat histImg = Mat::zeros(rows*scaleX, cols*scaleY, CV_8UC3);
	//for each bin
	for(int i=0;i<cols-1;i++) {
		float histValue = hist.at<float>(i,0);
		float nextValue = hist.at<float>(i+1,0);
		Point pt1 = Point(i*scaleX, rows*scaleY);
		Point pt2 = Point(i*scaleX+scaleX, rows*scaleY);
		Point pt3 = Point(i*scaleX+scaleX, (rows-nextValue*rows/maxVal)*scaleY);
		Point pt4 = Point(i*scaleX, (rows-nextValue*rows/maxVal)*scaleY);

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

	for (int i = 0; i < channels.size(); i++) {
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