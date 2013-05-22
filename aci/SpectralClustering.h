#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>
#include <queue>

#include "GraphSpectra.h"
#include "WeightedGraph.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

class SimilarityGraphRepresentation {
public:
	/**
	 * Compute a similarity graph out of a similarity matrix or function.
	 *
	 * @param simFunc input similarity matrix or function.
	 * @param graph output similarity graph.
	 */
	virtual void operator() (const MatrixXd &samples, double (*simFunc)(const VectorXd&, const VectorXd&), WeightedGraph &graph) const = 0;
};


/**
 * Spectral clustering of a set of samples by the k smallest eigenvectors 
 * for a specific matrix representation of a specific graph representation.
 *
 * @param samples samples to cluster.
 * @param simFunc similarity function.
 * @param graphRep graph representation to use for clustering.
 * @param matRep a sparse matrix graph representation. The representation is sparse
 * for efficient smallest eigenvectors computation.
 * @param k the number of clusters.
 * @param normalize indicates whether the spectral embedding coordinates should be
 * normalized. Useful in the case of normalized spectral clustering as defined by
 * Ng - Jordan - Weiss algorithm.
 * @param bidirectional indicates whether the graph representation is bidirectional
 * or not.
 */
void spectralClustering(const MatrixXd &samples, double (*simFunc)(const VectorXd&, const VectorXd&), SimilarityGraphRepresentation &graphRep, SparseRepresentation matRep, int k, VectorXi &classLabels, bool normalize = false, bool symmetric = true);

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
 * @param bidirectional indicates whether the graph representation is bidirectional
 * or not.
 */
void spectralClustering(const WeightedGraph &simGraph, SparseRepresentation matRep, int k, VectorXi &classLabels, bool normalize = false, bool symmetric = true, bool bidirectional = false);

/**
 * Neighborhood graph representation of a similarity matrix. Each vertex in the graph
 * is connected by an unweighted edge to each of its numbers closer than a given radius.
 */
class NeighborhoodGraph : SimilarityGraphRepresentation {
private:
	double radius;

public:
	NeighborhoodGraph(double radius);

	void operator() (const MatrixXd &samples, double (*simFunc)(const VectorXd&, const VectorXd&), WeightedGraph &graph) const;
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

	void operator() (const MatrixXd &samples, double (*simFunc)(const VectorXd&, const VectorXd&), WeightedGraph &graph) const;
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

	void operator() (const MatrixXd &samples, double (*simFunc)(const VectorXd&, const VectorXd&), WeightedGraph &graph) const;
};

/**
 * Complete graph representation of a similarity matrix. All pair of distinct
 * vertices are linked by exactly one edge weighted by similarity.
 */
class CompleteGraph : public SimilarityGraphRepresentation {
public:
	CompleteGraph();

	void operator() (const MatrixXd &samples, double (*simFunc)(const VectorXd&, const VectorXd&), WeightedGraph &graph) const;
};
