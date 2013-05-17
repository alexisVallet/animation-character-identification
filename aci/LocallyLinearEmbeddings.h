#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "GraphSpectra.h"

using namespace std;
using namespace cv;
using namespace Eigen;

/**
 * Computes locally linear embeddings into a smaller dimensional space from
 * samples, assuming those samples lie on a outDim-dimensional manifold.
 *
 * @param samples n by m matrix containing samples as row vectors.
 * @param outDim dimension of the embedded vectors, must be smaller than m .
 * @param embedding n by outdim output matrix containing the embeddings.
 * @param k number of nearest neighbors to consider for approximation of
 * each sample.
 */
void locallyLinearEmbeddings(const Mat_<float> &samples, int outDim, Mat_<float> &embeddings, int k);
