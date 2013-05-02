/** @file */
#pragma once

#include <opencv2/opencv.hpp>

#include "LabeledGraph.hpp"
#include "DisjointSet.hpp"

using namespace cv;

class DisjointSetForest;

/**
 * Euclidean distance between two matrices seen s vectors.
 */
double euclidDistance(const Mat &h1, const Mat &h2);

/**
 * Dot product between 2 color histograms seen as vectors.
 */
double dotProductKernel(const Mat &h1, const Mat &h2);

/**
 * Gaussian kernel between 2 color histograms.
 */
double gaussianKernel(float sigma, const Mat &h1, const Mat &h2);

/**
 * Weighted khi² (X²) kernel between two color histograms seen as
 * distributions.
 */
double khi2Kernel(int binsPerChannel, float lambda, float mu, float gamma, int area1, const Mat &h1, int area2, const Mat &h2);

/**
 * Weighs the edges of a graph by the result of a kernel function on the source
 * and destination vertices. Runs in O(m) where m is the number of edges in the graph
 * assuming the kernel function runs in O(1).
 *
 * @param kernel kernel function to weigh edges by.
 * @param unweightedGraph graph to compute edges from.
 * @return graph identical to the input graph with edges weighted by the kernel function.
 */
template < typename T >
WeightedGraph weighEdgesByKernel(double (*kernel)(const T &l1, const T &l2), LabeledGraph<T> unweightedGraph) {
	WeightedGraph weighted(unweightedGraph.numberOfVertices());

	for (int i = 0; i < unweightedGraph.getEdges().size(); i++) {
		Edge edge = unweightedGraph.getEdges()[i];
		double weight = kernel(unweightedGraph.getLabel(edge.source), unweightedGraph.getLabel(edge.destination));

		weighted.addEdge(edge.source, edge.destination, weight);
	}

	return weighted;
}
