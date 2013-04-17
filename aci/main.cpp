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
#include "IsoperimetricGraphPartitioning.h"
#include "GraphSpectraTest.h"
#include "IsoperimetricGraphPartitioningTest.h"

#define TEST true
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
	cout<<"computing grid graph"<<endl;
	WeightedGraph basicGraph = gridGraph(smoothed, CONNECTIVITY, mask, true);
	vector<int> vertexMap;
	cout<<"removing isolated vertices"<<endl;
	WeightedGraph connected = removeIsolatedVertices(basicGraph, vertexMap);
	cout<<"computing segmentation"<<endl;
	DisjointSetForest segmentationConn = isoperimetricGraphPartitioning(basicGraph, 0.25);
	cout<<"adding isolated vertices back"<<endl;
	DisjointSetForest segmentation = addIsolatedVertices(basicGraph, segmentationConn, vertexMap);
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

void computeRates(
	vector<LabeledGraph<Mat> > graphs,
	Mat classes,
	vector<pair<string, TrainableStatModel*> > models, 
	vector<pair<string, MatKernel> > kernels, 
	vector<pair<string, MatrixRepresentation> > representations) {
	for (int i = 0; i < (int)models.size(); i++) {
		cout<<models[i].first<<endl;
		for (int j = 0; j < (int)kernels.size(); j++) {
			cout<<kernels[j].first<<endl;
			for (int k = 0; k < (int)representations.size(); k++) {
				cout<<representations[k].first<<endl;
				SpectrumDistanceClassifier classifier(kernels[j].second, models[i].second, representations[k].second, EIG_MU);
				float rate = classifier.leaveOneOutRecognitionRate(graphs, classes);

				cout<<"rate = "<<rate<<endl;;
			}
		}
	}
}

int main(int argc, char** argv) {
	if (TEST) {
		testGraphSpectra();
		testIsoperimetricGraphPartitioning();
	} else {
		// loads the dataset
		char *folder = "C:\\Users\\Vallet\\Documents\\Dev\\animation-character-identification\\test\\dataset\\";
		char *names[] = {"amuro", "asuka", "char", "chirno", "conan", "jigen", "kouji", "lupin", "majin", "miku", "ray", "rufy"};
		vector<Mat> dataSet;
		Mat classes;

		loadDataSet(folder, names, 12, 5, dataSet, classes);

		// compute segmentation graphs
		vector<LabeledGraph<Mat> > graphs;

		for (int i = 0; i < (int)dataSet.size(); i++) {
			graphs.push_back(computeGraphFrom(dataSet[i]));
		}

		KNearestModel knnModel;
		BayesModel bayesModel;

		vector<pair<string, TrainableStatModel*> > models;

		models.push_back(pair<string,TrainableStatModel*>("Nearest neighbor", &knnModel));
		//models.push_back(pair<string,TrainableStatModel*>("Bayes", &bayesModel));
	
		vector<pair<string, MatKernel> > kernels;

		kernels.push_back(pair<string, MatKernel>("Dot product", dotProductKernel));
		kernels.push_back(pair<string, MatKernel>("Gaussian kernel", gaussianKernel_));
		kernels.push_back(pair<string, MatKernel>("Khi2 kernel", khi2Kernel_));

		vector<pair<string, MatrixRepresentation> > representations;

		representations.push_back(pair<string, MatrixRepresentation>("Combinatorial Laplacian", laplacian));
		representations.push_back(pair<string, MatrixRepresentation>("Normalized Laplacian", normalizedLaplacian));

		computeRates(graphs, classes, models, kernels, representations);
	}

	return 0;
}
