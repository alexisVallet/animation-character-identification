#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost\graph\boyer_myrvold_planar_test.hpp>
#include "DisjointSet.hpp"
#include "WeightedGraph.hpp"
#include "Felzenszwalb.hpp"
#include "SegmentationGraph.hpp"
#include "TreeWalkKernel.hpp"
#include "GraphSpectra.h"
#include "SpectrumDistanceClassifier.h"
#include "Kernels.h"

#define DEBUG false
#define BLUR_SIGMA 0.8
#define CONNECTIVITY CONNECTIVITY_4
#define MAX_SEGMENTS 100
#define BINS_PER_CHANNEL 16
#define EIG_MU 1
#define GAUSS_SIGMA 100
#define KHI_MU 0.01
#define KHI_LAMBDA 0.75

using namespace std;
using namespace cv;

LabeledGraph<Mat> computeGraphFrom(Mat &image) {
	// remove the alpha channel, turning to black transparent pixels.
	Mat_<float> mask(image.rows, image.cols);
	vector<Mat> channels(4);

	split(image, channels);

	mask = channels[3] / 255;

	vector<Mat_<uchar> > rgbChannels(3);

	for (int i = 0; i < 3; i++) {
		rgbChannels[i] = channels[i];
	}

	Mat_<Vec<uchar, 3> > rgbImage;

	merge(rgbChannels, rgbImage);
	// filter the image for better segmentation
	Mat_<Vec<uchar,3> > smoothed;

	GaussianBlur(rgbImage, smoothed, Size(0,0), BLUR_SIGMA);

	// segment the image using Felzenszwalb's method
	WeightedGraph basicGraph = gridGraph(smoothed, CONNECTIVITY, mask);
	DisjointSetForest segmentation = felzenszwalbSegment(
		min(rgbImage.rows,rgbImage.cols),
		basicGraph,
		(rgbImage.rows * rgbImage.cols) / MAX_SEGMENTS,
		mask);
	LabeledGraph<Mat> segGraph = segmentationGraph<Mat>(
		smoothed,
		segmentation,
		basicGraph);
	colorHistogramLabels(smoothed, segmentation, segGraph, BINS_PER_CHANNEL);
	if (DEBUG) {
		Mat regionImage = segmentation.toRegionImage(smoothed);
		segGraph.drawGraph(segmentCenters(smoothed, segmentation), regionImage);
		imshow("segmentation graph", regionImage);
		waitKey(0);
	}

	return segGraph;
}

static double gaussianKernel_(const Mat &h1, const Mat &h2) {
	return gaussianKernel(GAUSS_SIGMA, h1, h2);
}

static double khi2Kernel_(const Mat &h1, const Mat &h2) {
	return khi2Kernel(BINS_PER_CHANNEL, KHI_LAMBDA, KHI_MU, 1, 1, h1, 1, h2);
}

int main(int argc, char** argv) {
	// loads the dataset
	char *folder = "C:\\Users\\Vallet\\Documents\\Dev\\animation-character-identification\\test\\dataset\\";
	char *names[] = {"amuro", "asuka", "char", "chirno", "conan", "jigen", "kouji", "lupin", "majin", "miku", "ray", "rufy"};
	vector<Mat> dataSet;
	Mat classes;

	loadDataSet(folder, names, 12, 5, dataSet, classes);

	cout<<classes<<endl;
	// compute segmentation graphs
	vector<LabeledGraph<Mat> > graphs;

	for (int i = 0; i < dataSet.size(); i++) {
		graphs.push_back(computeGraphFrom(dataSet[i]));
	}

	KNearestModel model;
	SpectrumDistanceClassifier dpClassifier(dotProductKernel, &model, laplacian, EIG_MU);
	float dpRate = dpClassifier.leaveOneOutRecognitionRate(graphs, classes);
	SpectrumDistanceClassifier gaussClassifier(gaussianKernel_, &model, laplacian, EIG_MU);
	float gaussRate = gaussClassifier.leaveOneOutRecognitionRate(graphs, classes);
	SpectrumDistanceClassifier khi2Classifier(khi2Kernel_, &model, laplacian, EIG_MU);
	float khi2Rate = khi2Classifier.leaveOneOutRecognitionRate(graphs, classes);

	cout<<"Dot product: "<<dpRate<<endl<<"Gaussian kernel: "<<gaussRate<<endl<<"Khi2 kernel: "
<<khi2Rate<<endl;

	return 0;
}
