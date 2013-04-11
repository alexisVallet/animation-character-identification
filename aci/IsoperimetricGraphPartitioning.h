#pragma once

#include "WeightedGraph.hpp"
#include "DisjointSet.hpp"
#include "GraphSpectra.h"

using namespace cv;
using namespace std;

/**
 * Removes isolated vertices from the graph, generated from the background for
 * instance, for post-processing to the isoperimetric graph partitioning algorithm.
 * Optionally, returns a vector mapping vertices from graph to the result graph so
 * the transformation can be undone. Assumes a bidirectional representation for the graph.
 *
 * @param graph the graph to remove isolated vertices from.
 * @param vertexMap a vector of size n where n is the number of vertices in graph. Will
 * be populated with the new index for each vertex, or -1 if it was removed.
 */
WeightedGraph removeIsolatedVertices(WeightedGraph &graph, vector<int> &vertexMap = vector<int>());

/**
 * Segments a graph using the isoperimetric algorithm by Grady and Schwarz.
 *
 * @param graph the graph to partition.
 * @param stop maximum isoperimetric ratio for segments.
 * @return a partition of the graph.
 */
DisjointSetForest isoperimetricGraphPartitioning(const WeightedGraph &graph, double stop);

/**
 * Solves the linear system Ax = b for x where A is a n by n symmetric positive definite matrix,
 * and b and x are n by 1 column vectors. Takes an approximate initial solution for x or the 0 vector.
 */
Mat_<double> conjugateGradient(SparseMat_<double> &A, Mat_<double> &b, Mat_<double> &x);
