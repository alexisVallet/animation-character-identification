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
#define BINS_PER_CHANNEL 5
#define TREE_WALK_ARITY 2
#define TREE_WALK_DEPTH 2
#define LAMBDA 1
#define MU 1

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
	WeightedGraph nnGraph = gridGraph(smoothed, CONNECTIVITY);
	DisjointSetForest segmentation = felzenszwalbSegment(k, nnGraph, minCompSize);
	histogramGraph = segmentationGraph<Mat>(smoothed, segmentation, nnGraph);
	cout<<histogramGraph.numberOfVertices()<<" vertices"<<endl;
	cout<<histogramGraph.getEdges().size()<<" edges"<<endl;
	cout<<histogramGraph<<endl;
	Mat_<Vec<uchar,3> > segmentationImage = segmentation.toRegionImage(image);
	cout<<"computing boost graph"<<endl;
	graph_t
		boostGraph = histogramGraph.toBoostGraph();
	//showBoostGraph(histogramGraph, boostGraph);
	cout<<"computing planarity"<<endl;
	//bool isPlanar = boyer_myrvold_planarity_test(boyer_myrvold_params::graph = boostGraph);
	//cout<<"Planarity: "<<isPlanar<<endl;

	//histogramGraph.drawGraphWithEmbedding(segmentCenters(image,segmentation), segmentationImage, boostGraph, embedding);
	histogramGraph.drawGraph(segmentCenters(image,segmentation),segmentationImage);
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

	Mat_<double> unnormalized = laplacian(graph1);
	Mat uEigenvectors;
	Mat uEigenvalues;

	eigen(unnormalized, uEigenvalues, uEigenvectors);

	cout<<"Laplacian eigenvalues: "<<endl<<uEigenvalues<<endl;

	Mat_<double> normalized = normalizedLaplacian(graph1);
	Mat nEigenvectors;
	Mat nEigenvalues;

	eigen(normalized, nEigenvalues, nEigenvectors);

	cout<<"Normalized laplacian eigenvalues: "<<endl<<nEigenvalues<<endl;

	return 0;
}
