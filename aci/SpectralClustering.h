/** @file */
#pragma once

#include <boost/variant.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>
#include <queue>

#include "GraphSpectra.h"
#include "WeightedGraph.hpp"
#include "spectral-graph-theory/SimilarityGraphs.h"

using namespace std;
using namespace cv;
using namespace Eigen;

/**
 * Spectral clustering of a set of samples by the k smallest eigenvectors 
 * for a specific matrix representation of a specific graph representation.
 *
 * @param similarity similarity matrix.
 * @param graphRep graph representation to use for clustering.
 * @param matRep a sparse matrix graph representation. The representation is sparse
 * for efficient smallest eigenvectors computation.
 * @param k the number of clusters.
 * @param classLabels output vector of class labels corresponding to each sample.
 * @param normalize indicates whether the spectral embedding coordinates should be
 * normalized. Useful in the case of normalized spectral clustering as defined by
 * Ng - Jordan - Weiss algorithm.
 * @param bidirectional indicates whether the graph representation is bidirectional
 * or not.
 */
void spectralClustering(SimilarityMatrix &similarity, SimilarityGraphRepresentation &graphRep, SparseRepresentation matRep, int k, VectorXi &classLabels, bool normalize = false, bool symmetric = true);

/**
 * Self tuning spectral clustering as defined by (Zelnik-Manor and Perona, 2004).
 * Right now only uses local scaling, does not infer the number of clusters, and
 * uses K-means for the final step.
 *
 * @param samples samples to cluster as rows of a matrix.
 * @param nbClusters number of clusters.
 * @param classLabels output vector of class labels corresponding to each sample.
 */
//void selfTuningSpectralClustering(const MatrixXd &samples, int nbClusters, VectorXi &classLabels);

/**
 * Spectral embedding according to similarity matrix in k-dimensional space.
 */
void spectralEmbedding(SimilarityMatrix &similarity, SimilarityGraphRepresentation &graphRep, SparseRepresentation matRep, int k, MatrixXd &embeddings, bool normalize = false, bool symmetric = true);

