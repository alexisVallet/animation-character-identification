#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>

#include "GraphSpectra.h"
#include "WeightedGraph.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

/**
 * General spectral clustering routine, clustering an arbitrary weighted graph
 * by the k smallest (by their corresponding eigenvalues) eigenvectors of a
 * specific matrix representation.
 *
 * @param simGraph similarity graph, meaning weights indicate how similar vertices
 * are.
 * @param matRep a sparse matrix graph representation. The representation is sparse
 * for efficient smallest eigenvectors computation.
 * @param k the number of clusters.
 * @param normalize indicates whether the spectral embedding coordinates should be
 * normalized. Useful in the case of normalized spectral clustering as defined by
 * Ng - Jordan - Weiss algorithm.
 */
void spectralClustering(const WeightedGraph &simGraph, SparseRepresentation matRep, int k, VectorXi &classLabels, bool normalize = false, bool symmetric = true);
