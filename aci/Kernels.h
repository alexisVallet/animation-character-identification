/** @file */
#pragma once

#include <opencv2/opencv.hpp>

#include "LabeledGraph.hpp"
#include "DisjointSet.hpp"
#include "Utils.hpp"
#include "GraphPartitions.h"
#include "SegmentAttributes.h"

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
 * Gaussian kernel between 2 vectors or matrices seen as vectors.
 */
double gaussianKernel(float mu, float sigma, const Mat &h1, const Mat &h2);

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
template < typename _Tp, int m, int n >
WeightedGraph weighEdgesByKernel(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, const MatKernel<_Tp, m, n> &kernel, const LabeledGraph<Matx<_Tp, m, n> > &unweightedGraph) {
	LabeledGraph<Matx<_Tp, m, n> > labeled;
	kernel.getLabeling()(image, mask, segmentation, unweightedGraph, labeled);

	WeightedGraph weighted(unweightedGraph.numberOfVertices());

	for (int i = 0; i < (int)labeled.getEdges().size(); i++) {
		Edge edge = labeled.getEdges()[i];
		float weight = (float)kernel(labeled.getLabel(edge.source), labeled.getLabel(edge.destination));
		weighted.addEdge(edge.source, edge.destination, weight);
	}

	return weighted;
}

/**
 * Functor for computing similarity between 2 neighboring segments.
 */
class CompoundGaussianKernel : public MatKernel<float, 8, 1> {
private:
	double alphaC;
	double alphaX;
	double alphaS;

public:
	CompoundGaussianKernel(double alphaC, double alphaX, double alphaS);

	Labeling<float,8,1>::type getLabeling() const;

	double operator() (const Matx<float,8,1> &h1, const Matx<float,8,1> &h2) const;
};
