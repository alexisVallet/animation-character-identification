#pragma once

#include <Eigen/Dense>
#include <vector>
#include <iostream>

#include "../WeightedGraph.hpp"
#include "../GraphSpectra.h"

using namespace Eigen;
using namespace std;

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
 * Efficiently wraps around another similarity matrix, hiding some rows
 * and columns. Effectively models a principal submatrix of a square
 * matrix.
 */
class MaskedSimilarityMatrix : public SimilarityMatrix {
private:
	const SimilarityMatrix const *internalMatrix;
	const vector<int> const *indexes;

public:
	MaskedSimilarityMatrix(const SimilarityMatrix const *internalMatrix, const vector<int> const *indexes);
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
 * Computes the k nearest neighbors of each sample in a similarity matrix.
 *
 * @param similarity matrix of similarities between samples.
 * @param k number of nearest neighbors to compute.
 * @return n by k array containing the indexes of the k nearest neighbor
 * of each n sample.
 */
vector<vector<int> > similarityKNN(SimilarityMatrix &similarity, int k);

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

/**
 * Transforms a similarity graph representation into a smaller one masking
 * some points.
 */
class MaskedGraph : public SimilarityGraphRepresentation {
private:
	SimilarityGraphRepresentation *internalRep;
	vector<bool> mask;

public:
	MaskedGraph(vector<bool> mask, SimilarityGraphRepresentation *internalRep);
	void operator() (SimilarityMatrix &similarity, WeightedGraph &graph) const;
};
