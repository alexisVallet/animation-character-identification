#pragma once

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/Dense>

#include "WeightedGraph.hpp"
#include "DisjointSet.hpp"
#include "GraphSpectra.h"

using namespace std;
using namespace Eigen;

/**
 * Partitions the graph using the normalized cuts method.
 *
 * @param graph graph to partition.
 * @param stop minimal normalized ratio acting a stopping criteria.
 */
DisjointSetForest normalizedCuts(WeightedGraph &graph, double stop);
