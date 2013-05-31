/** @file */
#pragma once

#include <opencv2/opencv.hpp>
#include <boost/graph/boyer_myrvold_planar_test.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <vector>
#include <iostream>
#include <cmath>

#include "LabeledGraph.hpp"
#include "DisjointSet.hpp"
#include "Kernels.h"
#include "GraphPartitions.h"

using namespace std;
using namespace cv;
using namespace boost;

static int current(int iteration) {
  return iteration % 2;
}

static int previous(int iteration) {
  return !current(iteration);
}

/**
 * Function class to compare neighbors of a vertex in circular order.
 */
class CircularCompare {
private:
	int src;
	const vector<Vec2f> *centers;

public:
	/**
	 * Initialize the comparison function with a given source vertex
	 */
	CircularCompare(int src, const vector<Vec2f> *centers)
		: src(src), centers(centers)
	{

	}

	bool operator() (const int &dst1, const int &dst2) {
		Vec2f 
			v1 = (*centers)[dst1] - (*centers)[src], 
			v2 = (*centers)[dst2] - (*centers)[src];
		double angle1 = atan2(v1[0], v1[1]);
		double angle2 = atan2(v2[0], v2[1]);

		return angle1 < angle2;
	}
};

static void computeNeighbors(const WeightedGraph &graph, const vector<Vec2f> &embedding, vector<vector<int> > &circNeighbors) {
	circNeighbors = vector<vector<int> >(graph.numberOfVertices());

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		circNeighbors[i] = vector<int>();
		circNeighbors[i].reserve(graph.getAdjacencyList(i).size());

		for (int j = 0; j < graph.getAdjacencyList(i).size(); j++) {
			circNeighbors[i].push_back(graph.getAdjacencyList(i)[j].destination);
		}

		CircularCompare comp(i, &embedding);

		sort(circNeighbors[i].begin(), circNeighbors[i].end(), comp);
	}
}

/**
 * Computes the tree walk kernel between two labelled graphs given
 * a basis kernel function. It is assumed that each graph is undirected,
 * with each edge duplicated in both the source and destination's
 * adjacency list.
 *
 * Unlike Bach and Harchaoui's initial formulation, this does not assume
 * the graphs are planar. Indeed, segmentation methods considering pixels
 * in 8-connexity neighborhood, of in feature-space neighborhood may not
 * result in a planar segmentation graph at all. If it is planar, the embedding
 * can be computed in O(n) using the Boyer-Myrvold algorithm for instance, but
 * such an embedding may not correspond to the spatial layout of the segments in
 * the image.
 *
 * This implementation therefore uses a user defined embedding for each graph,
 * and computes the circular ordering from this embedding.
 *
 * @param basisKernel basis kernel function, taking 2 labels as parameters,
 * with the associated area for each segment.
 * @param segmentation1 segmentation corresponding to graph1
 * @param graph1 first segmentation graph
 * @param embedding1 plane (not necessarily planar) embedding for vertices in graph1
 * @param segmentation2 segmentation corresponding to graph2
 * @param graph2 second segmentation graph
 * @param embedding2 plane (not necessarily planar) embedding for vertices in graph2
 */
template < typename _Tp, int m, int n >
double treeWalkKernel(
	const MatKernel<_Tp, m, n> &basisKernel, 
	int depth, 
	int arity,
	const LabeledGraph<Matx<_Tp, m, n> > &graph1,
	const vector<Vec2f> &embedding1,
	const LabeledGraph<Matx<_Tp, m, n> > &graph2,
	const vector<Vec2f> &embedding2) {
	// basisKernels contains the basis kernel for each pair of vertices in
	// the two graphs, computed once and for all.
	Mat_<double> basisKernels = Mat_<double>::zeros(graph1.numberOfVertices(), graph2.numberOfVertices());

	// depthKernels contains the current iteration's results as well as the previous
	// iteration's results, swapped at each iteration in a "ping-pong" fashion.
	Mat_<double> depthKernels[2] = {
		Mat_<double>::zeros(graph1.numberOfVertices(), graph2.numberOfVertices()),
		Mat_<double>::zeros(graph1.numberOfVertices(), graph2.numberOfVertices())
	};

	// neighbors for each vertex in each graph in the circular order given
	// by the embeddings.
	vector<vector<int> > 
		circNeighbors1, 
		circNeighbors2;

	//initializes neighbors in circular order
	computeNeighbors(graph1, embedding1, circNeighbors1);
	computeNeighbors(graph2, embedding2, circNeighbors2);

	// initializes basis kernels
	for (int v1 = 0; v1 < graph1.numberOfVertices(); v1++) {
		for (int v2 = 0; v2 < graph2.numberOfVertices(); v2++) {
			basisKernels(v1,v2) = basisKernel(
				graph1.getLabel(v1),
				graph2.getLabel(v2));
			depthKernels[previous(0)](v1,v2) = basisKernels(v1,v2);
		}
	}

	// computes kernels for each depth up to the maximum depth
	for (int d = 0; d < depth; d++) {
		for (int v1 = 0; v1 < graph1.numberOfVertices(); v1++) {
			for (int v2 = 0; v2 < graph2.numberOfVertices(); v2++) {
				double sum = 0;

				// summing for neighbor intervals of cardinal lower
				// than arity
				for (int iSize = 1; iSize <= arity; iSize++) {
					// if the size is greater than either number of
					// neighbors, there are no such neighbor intervals
					if (iSize > (int)circNeighbors1[v1].size() || iSize > (int)circNeighbors2[v2].size()) {
						break;
					}
					// for each neighbor interval i (resp j) of v1 (resp v2)
					for (int i = 0; i < (int)circNeighbors1[v1].size(); i++) {
						for (int j = 0; j < (int)circNeighbors2[v2].size(); j++) {
							double product = 1;
							// for each neighbor r (resp s) of v1 (resp v2) in i (resp j)
							for (int r = i; r < i + iSize; r++) {
								for (int s = j; s < j + iSize; s++) {
									int neighbor1 = circNeighbors1[v1][r % circNeighbors1[v1].size()];
									int neighbor2 = circNeighbors2[v2][s % circNeighbors2[v2].size()];
									
									product *= depthKernels[previous(d)](neighbor1, neighbor2);
								}
							}

							sum += product;
						}
					}
				}
				
				double res = basisKernels(v1,v2) * sum;

				depthKernels[current(d)](v1,v2) = res;
			}
		}
	}

	Scalar result = sum(depthKernels[previous(depth)]);
	
	// sum and return the results of the last iteration
	return result[0];
}
