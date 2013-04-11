#pragma once

#include "WeightedGraph.hpp"
#include "DisjointSet.hpp"
#include "GraphSpectra.h"

using namespace cv;
using namespace std;

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
SparseMat_<double> conjugateGradient(SparseMat_<double> A, SparseMat_<double> b, SparseMat_<double> x);