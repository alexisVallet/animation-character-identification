#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost\graph\boyer_myrvold_planar_test.hpp>
#include "DisjointSet.hpp"
#include "WeightedGraph.hpp"
#include "Felzenszwalb.hpp"
#include "SegmentationGraph.hpp"
#include "TreeWalkKernel.hpp"
#include "GraphSpectra.h"

#define SIGMA 0.8
#define CONNECTIVITY CONNECTIVITY_4
#define KNN 10
#define MAX_SEGMENTS 100
#define BINS_PER_CHANNEL 16
#define TREE_WALK_ARITY 1
#define TREE_WALK_DEPTH 0
#define LAMBDA 0.75
#define MU 0.01
#define GAMMA 0.8

using namespace std;
using namespace cv;

void showBoostGraph(WeightedGraph graph, graph_t bGraph) {
	cout<<"pouet"<<endl;
	for (int i = 0; i < graph.numberOfVertices(); i++) {
		cout<<i<<" : [";
		typedef graph_traits<graph_t>::adjacency_iterator adjacency_iterator;
		pair<adjacency_iterator,adjacency_iterator> neighbors = adjacent_vertices(i, bGraph);
		adjacency_iterator it;

		for (it = neighbors.first; it != neighbors.second; it++) {
			int neighbor = get(vertex_index, bGraph, *it);

			cout<<neighbor<<", ";
		}
		cout<<"]"<<endl;
	}
}

void computeLabeledGraphs(char* filename, LabeledGraph<Mat> &segGraph, DisjointSetForest &segmentation) {
	Mat_<Vec<uchar,3> > imageRGB = imread(filename);
	if (imageRGB.cols == 0) {
	    cout<<"Image "<<filename<<" not found."<<endl;
	    exit(0);
	}
	int minCompSize = imageRGB.rows * imageRGB.cols / MAX_SEGMENTS;
	int k = min(imageRGB.rows, imageRGB.cols) / 2;
	Mat_<Vec<uchar,3> > image;
	cvtColor(imageRGB, image, CV_RGB2Lab);
	imshow("lab",image);
	Mat_<Vec<uchar,3> > smoothed;
	GaussianBlur(image, smoothed, Size(0,0), SIGMA);
	WeightedGraph nnGraph = gridGraph(smoothed, CONNECTIVITY);
	segmentation = felzenszwalbSegment(k, nnGraph, minCompSize);
	segGraph = segmentationGraph<Mat>(smoothed, segmentation, nnGraph);
	cout<<segGraph.numberOfVertices()<<" vertices"<<endl;
	cout<<segGraph.getEdges().size()<<" edges"<<endl;
	cout<<segGraph<<endl;
	Mat_<Vec<uchar,3> > segmentationImage = segmentation.toRegionImage(image);
	graph_t
		boostGraph = segGraph.toBoostGraph();
	embedding_storage_t
		embedding_storage(num_vertices(boostGraph));
	embedding_t
		embedding(embedding_storage.begin(), get(vertex_index, boostGraph));
	bool isPlanar = boyer_myrvold_planarity_test(
		boyer_myrvold_params::graph = boostGraph,
		boyer_myrvold_params::embedding = embedding);
	cout<<"Planarity: "<<isPlanar<<endl;
	/*if (isPlanar) {
		segGraph.drawGraphWithEmbedding(segmentCenters(smoothed, segmentation), segmentationImage, boostGraph, embedding);
	}*/
	segGraph.drawGraph(segmentCenters(smoothed, segmentation), segmentationImage);
	imshow("segmentation graph", segmentationImage);
	waitKey(0);
	colorHistogramLabels(smoothed, segmentation, segGraph, BINS_PER_CHANNEL);
}

double basisKernel(int area1, const Mat &c1, int area2, const Mat &c2) {
	return khi2Kernel(BINS_PER_CHANNEL, LAMBDA, MU, GAMMA, area1, c1, area2, c2);
}

int main(int argc, char** argv) {
    if (argc < 2) {
		cout<<"Please enter an image filename to process."<<endl;
		return 0;
	}

	LabeledGraph<Mat> graph1;
	DisjointSetForest segmentation1;
	
	computeLabeledGraphs(argv[1], graph1, segmentation1);

/*	Mat_<double> unnormalized = laplacian(graph1);
	Mat uEigenvectors;
	Mat uEigenvalues;

	eigen(unnormalized, uEigenvalues, uEigenvectors);

	cout<<"Laplacian eigenvalues: "<<endl<<uEigenvalues<<endl;

	Mat_<double> normalized = normalizedLaplacian(graph1);
	Mat nEigenvectors;
	Mat nEigenvalues;

	eigen(normalized, nEigenvalues, nEigenvectors);

	cout<<"Normalized laplacian eigenvalues: "<<endl<<nEigenvalues<<endl; */

	if (argc >= 3) {
		LabeledGraph<Mat> graph2;
		DisjointSetForest segmentation2;

		computeLabeledGraphs(argv[2], graph2, segmentation2);

		double twk = treeWalkKernel<Mat>(basisKernel, TREE_WALK_DEPTH, TREE_WALK_ARITY, segmentation1, graph1, segmentation2, graph2);

		cout<<"Tree walk kernel between the 2 segmentation graphs is "<<twk<<endl;
	}

	return 0;
}
