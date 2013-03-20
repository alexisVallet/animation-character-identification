#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost\graph\boyer_myrvold_planar_test.hpp>
#include "DisjointSet.hpp"
#include "WeightedGraph.hpp"
#include "Felzenszwalb.hpp"
#include "SegmentationGraph.hpp"
#include "TreeWalkKernel.hpp"

#define SIGMA 0.8
#define CONNECTIVITY CONNECTIVITY_4
#define MAX_SEGMENTS 50
#define BINS_PER_CHANNEL 5
#define TREE_WALK_ARITY 2
#define TREE_WALK_DEPTH 2
#define LAMBDA 0.75
#define MU 1

using namespace std;
using namespace cv;

void computeLabeledGraphs(char* filename, LabeledGraph<Mat> &histogramGraph) {
	Mat_<Vec<uchar,3> > imageRGB = imread(filename);
	if (imageRGB.cols == 0) {
	    cout<<"Image "<<filename<<" not found."<<endl;
	    exit(0);
	}
	int minCompSize = imageRGB.rows * imageRGB.cols / MAX_SEGMENTS;
	int k = min(imageRGB.rows, imageRGB.cols) / 2;
	Mat_<Vec<uchar,3> > image;
	cvtColor(imageRGB, image, CV_RGB2Lab);
	Mat_<Vec<uchar,3> > smoothed;
	GaussianBlur(image, smoothed, Size(0,0), SIGMA);
	WeightedGraph grid = gridGraph(smoothed, CONNECTIVITY);
	DisjointSetForest segmentation = felzenszwalbSegment(k, grid, minCompSize);
	histogramGraph = segmentationGraph<Mat>(smoothed, segmentation, grid);
	Mat_<Vec<uchar,3> > segmentationImage = segmentation.toRegionImage(image);
	adjacency_list<vecS,vecS,bidirectionalS,property<vertex_index_t, int> > 
    boostGraph = histogramGraph.toBoostGraph();
	bool isPlanar = boyer_myrvold_planarity_test(boostGraph);
	cout<<"Planarity: "<<isPlanar<<endl;
	assert(isPlanar);
	histogramGraph.drawGraph(segmentCenters(image,segmentation), segmentationImage);
	imshow("segmentation graph", segmentationImage);
	waitKey(0);
	colorHistogramLabels(smoothed, segmentation, histogramGraph, BINS_PER_CHANNEL);
}

double basisKernel(const Mat& h1, const Mat& h2) {
	return khi2Kernel(BINS_PER_CHANNEL, LAMBDA, MU, h1, h2);
}

int main(int argc, char** argv) {
    if (argc < 2) {
		cout<<"Please enter an image filename to process."<<endl;
		return 0;
	}

	LabeledGraph<Mat> graph1;
	
	computeLabeledGraphs(argv[1], graph1);


	if (argc >= 3) {
		LabeledGraph<Mat> graph2;

		computeLabeledGraphs(argv[2], graph2);

		double kernelResult = treeWalkKernel<Mat>(basisKernel, TREE_WALK_DEPTH, TREE_WALK_ARITY, graph1, graph2);

		cout<<"Tree walk kernel between "<<argv[1]<<" and "<<argv[2]<<" is "<<kernelResult<<endl;
	}

	return 0;
}