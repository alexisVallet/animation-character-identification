#pragma once

#include <opencv2\opencv.hpp>
#include <complex>

#include "WeightedGraph.hpp"

using namespace std;
using namespace cv;

typedef Mat_<double> (*MatrixRepresentation)(const WeightedGraph&);

/**
 * Computes the unnormalized laplacian matrix L of a weighted graph, defined
 * by:
 *
 * L(u,v) = dv - w(v,v)	if u = v
 *          -w(u,v)		if u and v are adjacent
 *			0			otherwise
 *
 * Assumes the graph represents each edge only once, ie. that the
 * adjacency lists are not bidirectional.
 *
 * @param graph graph to compute the laplacian from.
 * @return the laplacian matrix of the weighted graph.
 */
Mat_<double> laplacian(const WeightedGraph &graph);

/**
 * Computes the normalized laplacian NL matrix of a weighted graph, defined by:
 *
 * NL(u,v) = 1 - w(v,v)/dv			if u = v
 *			 -w(u,v)/sqrt(du*dv)	if u and v are adjacent
 *			0						otherwise
 * Assumes the graph represents each edge only once, ie. that the
 * adjacency lists are not bidirectional.
 *
 * @param graph graph to compute the laplacian from.
 * @return the normalized laplacian matrix of the weighted graph.
 */
Mat_<double> normalizedLaplacian(const WeightedGraph &graph);
