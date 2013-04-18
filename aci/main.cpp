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
#include "ImageGraphsTest.h"

#define TEST false
#define DEBUG true
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

LabeledGraph<Mat> computeGraphFrom(Mat_<Vec<uchar,3> > &rgbImage, Mat_<float> &mask) {
	// filter the image for better segmentation
	Mat_<Vec<uchar,3> > smoothed;

	GaussianBlur(rgbImage, smoothed, Size(0,0), BLUR_SIGMA);

	cout<<"computing grid graph"<<endl;
	// segment the image using Felzenszwalb's method
<<<<<<< HEAD
	cout<<"computing grid graph"<<endl;
	WeightedGraph basicGraph = gridGraph(smoothed, CONNECTIVITY, mask, true);
	vector<int> vertexMap;
	cout<<"removing isolated vertices"<<endl;
	WeightedGraph connected = removeIsolatedVertices(basicGraph, vertexMap);
	cout<<"computing segmentation"<<endl;
	DisjointSetForest segmentationConn = unconnectedIGP(connected, 0.25);
	cout<<"adding isolated vertices back"<<endl;
	DisjointSetForest segmentation = addIsolatedVertices(basicGraph, segmentationConn, vertexMap);
=======
	WeightedGraph basicGraph = gridGraph(smoothed, CONNECTIVITY, mask);
	cout<<"segmenting"<<endl;
	DisjointSetForest segmentation = felzenszwalbSegment(
		min(rgbImage.rows,rgbImage.cols),
		basicGraph,
		(rgbImage.rows * rgbImage.cols) / MAX_SEGMENTS,
		mask);
	cout<<"computing segmentation graph"<<endl;
>>>>>>> master
	LabeledGraph<Mat> segGraph = segmentationGraph<Mat>(
		smoothed,
		segmentation,
		basicGraph);
	cout<<"computing color histograms"<<endl;
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
<<<<<<< HEAD
	if (TEST) {
		testGraphSpectra();
		testImageGraphs();
		testIsoperimetricGraphPartitioning();
	} else {
		// loads the dataset
		char *folder = "C:\\Users\\Vallet\\Documents\\Dev\\animation-character-identification\\test\\dataset\\";
		char *names[] = {"amuro", "asuka", "char", "chirno", "conan", "jigen", "kouji", "lupin", "majin", "miku", "ray", "rufy"};
		vector<Mat> dataSet;
		Mat classes;
=======
	// loads the dataset
	cout<<"loading dataset"<<endl;
	char *folder = "../test/dataset/";
	char *names[] = {"amuro", "asuka", "char", "chirno", "conan", "jigen", "kouji", "lupin", "majin", "miku", "ray", "rufy"};
	vector<pair<Mat_<Vec<uchar,3> >,Mat_<float> > > dataSet;
	Mat classes;
>>>>>>> master

		loadDataSet(folder, names, 12, 5, dataSet, classes);

		// compute segmentation graphs
		vector<LabeledGraph<Mat> > graphs;

<<<<<<< HEAD
		for (int i = 0; i < (int)dataSet.size(); i++) {
			graphs.push_back(computeGraphFrom(dataSet[i]));
		}

		KNearestModel knnModel;
		BayesModel bayesModel;

		vector<pair<string, TrainableStatModel*> > models;
=======
	cout<<"computing segmentation graphs"<<endl;
	for (int i = 0; i < dataSet.size(); i++) {
		graphs.push_back(computeGraphFrom(dataSet[i].first, dataSet[i].second));
	}

	cout<<"measuring results"<<endl;
	KNearestModel model;
	SpectrumDistanceClassifier dpClassifier(dotProductKernel, &model, laplacian, EIG_MU);
	float dpRate = dpClassifier.leaveOneOutRecognitionRate(graphs, classes);
	SpectrumDistanceClassifier gaussClassifier(gaussianKernel_, &model, laplacian, EIG_MU);
	float gaussRate = gaussClassifier.leaveOneOutRecognitionRate(graphs, classes);
	SpectrumDistanceClassifier khi2Classifier(khi2Kernel_, &model, laplacian, EIG_MU);
	float khi2Rate = khi2Classifier.leaveOneOutRecognitionRate(graphs, classes);
>>>>>>> master

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
