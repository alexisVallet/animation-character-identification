/** @file */
#pragma once

#include <boost/variant.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>
#include <queue>

#include "GraphSpectra.h"
#include "WeightedGraph.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

/**
 * Abstract class for similarity matrices. Implemented in
 * dense matrix from by DenseSimilarityMatrix, but may
 * also be implemented using any other arbitrary computations
 * in the case where the order of the matrix is too large to
 * be stored in memory.
 */
class SimilarityMatrix {
public:
	/**
	 * Returns the matrix element at row i and column j.
	 *
	 * @param i row index.
	 * @param j column index.
	 * @return the coefficient at row i and column j.
	 */
	virtual double operator() (int i, int j) const = 0;
	virtual int rows() const = 0;
	virtual int cols() const = 0;
};

/**
 * Simple wrapper around Eigen's MatrixXd datatype.
 */
class DenseSimilarityMatrix : public SimilarityMatrix {
private:
	const MatrixXd *m;

public:
	DenseSimilarityMatrix(const MatrixXd *m);
	double operator() (int i, int j) const;
	int rows() const;
	int cols() const;
};

/**
 * Abstract class for graph representation function objects to be applied on 
 * data samples.
 */
class SimilarityGraphRepresentation {
public:
	/**
	 * Compute a similarity graph out of a similarity matrix or function.
	 *
	 * @param simFunc input similarity matrix or function.
	 * @param graph output similarity graph.
	 */
	virtual void operator() (SimilarityMatrix &similarity, WeightedGraph &graph) const = 0;
};

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

/**
 * Neighborhood graph representation of a similarity matrix. Each vertex in the graph
 * is connected by an unweighted edge to each of its neighbors closer than a given radius.
 */
class NeighborhoodGraph : SimilarityGraphRepresentation {
private:
	double radius;

public:
	NeighborhoodGraph(double radius);

	void operator() (SimilarityMatrix &similarity, WeightedGraph &graph) const;
};

/**
 * K-nearest neighbor graph representation of a similarity matrix. A pair of vertices
 * in the graph are connected to each other iff either of them is among the K-nearest
 * neighbors of the other. Edges are weighted by similarity.
 */
class KNearestGraph : public SimilarityGraphRepresentation {
private:
	int k;

public:
	KNearestGraph(int k);

	void operator() (SimilarityMatrix &similarity, WeightedGraph &graph) const;
};

/**
 * K-nearest neighbor graph representation of a similarity matrix. A pair of vertices
 * in the graph are connected to each other iff both of them are among the K-nearest
 * neighbors of the other. Edges are weighted by similarity.
 */
class MutualKNearestGraph : public SimilarityGraphRepresentation {
private:
	int k;

public:
	MutualKNearestGraph(int k);

	void operator() (SimilarityMatrix &similarity, WeightedGraph &graph) const;
};

/**
 * Complete graph representation of a similarity matrix. All pair of distinct
 * vertices are linked by exactly one edge weighted by similarity.
 */
class CompleteGraph : public SimilarityGraphRepresentation {
public:
	CompleteGraph();

	void operator() (SimilarityMatrix &similarity, WeightedGraph &graph) const;
};
