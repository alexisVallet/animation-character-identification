#include "PreProcessing.h"

#define DEBUG_PREPROCESSING false

static double constOne(const Mat& m1, const Mat& m2) {
	return 1;
}

static void removeSmallComponents(const Mat_<float> &mask, Mat_<float> &connectedMask) {
	Mat_<Vec3b> dummy(mask.rows, mask.cols);
	WeightedGraph grid = gridGraph(dummy, CONNECTIVITY_4, mask, constOne, true);
	vector<int> inCC;
	int nbCC;
	vector<WeightedGraph> components;
	vector<int> vertexIdx;

	connectedComponents(grid, inCC, &nbCC);
	inducedSubgraphs(grid, inCC, nbCC, vertexIdx, components);
	int largestIndex = max_element(components.begin(), components.end(), compareGraphSize) - components.begin();

	connectedMask = Mat_<float>(mask.rows, mask.cols);

	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (inCC[toRowMajor(mask.cols, j, i)] == largestIndex) {
				connectedMask(i,j) = 1;
			} else {
				connectedMask(i,j) = 0;
			}
		}
	}
}

void preProcessing(const Mat_<Vec3b> &rawImage, const Mat_<float> &rawMask, Mat_<Vec3f> &processedImage, Mat_<float> &processedMask, const Mat_<Vec3b> &manualSegmentation, Mat_<Vec3b> &processedSegmentation, int kuwaharaHalfsize, int maxNbPixels) {
	assert(kuwaharaHalfsize <= (numeric_limits<uchar>::max() - 1) / 2);

	Mat_<Vec3b> resized;
	Mat_<float> resizedMask;

	resizeImage(rawImage, rawMask, resized, resizedMask, maxNbPixels, manualSegmentation, processedSegmentation);

	removeSmallComponents(resizedMask, processedMask);

	Mat_<Vec3b> equalized;
	
	//equalizeColorHistogram(resized, processedMask, equalized);
	equalized = resized;

	Mat_<Vec3b> filtered;

	KuwaharaFilter(equalized, filtered, 2 * kuwaharaHalfsize + 1);

	Mat_<Vec3f> filteredFloat = Mat_<Vec3f>(filtered) / 255.;

	cvtColor(filteredFloat, processedImage, CV_BGR2Lab);

	if (DEBUG_PREPROCESSING) {
		imshow("raw", rawImage);
		imshow("resized", resized);
		imshow("equalized", equalized);
		imshow("filtered", filtered);
		imshow("filteredFloat", filteredFloat);
		vector<Mat_<float> > channels;

		split(processedImage, channels);

		imshow("L", channels[0]/100.);
		imshow("a", (channels[1] + 127.)/256.);
		imshow("b", (channels[2] + 127.)/256.);

		Mat_<Vec3f> hsv;

		cout<<"converting"<<endl;
		cvtColor(filteredFloat, hsv, CV_BGR2HSV);

		vector<Mat_<float> > hsvChannels;

		cout<<"splitting"<<endl;
		split(hsv, hsvChannels);

		cout<<"displaying"<<endl;
		imshow("hue", hsvChannels[0] / 360.);

		waitKey(0);
	}
}
