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
// isoperimetricGraphPartitioning(const WeightedGraph &graph, double stop);
