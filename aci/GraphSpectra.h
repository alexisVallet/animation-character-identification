/** @file */
#pragma once

#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>
#include <complex>
#include <limits>

#include "WeightedGraph.hpp"
#include "Utils.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

typedef Mat_<double> (*MatrixRepresentation)(const WeightedGraph&);
typedef MatrixXd (*DenseRepresentation)(const WeightedGraph&);
typedef Eigen::SparseMatrix<double> (*SparseRepresentation)(const WeightedGraph &, bool bidirectional);

/**
 * Computes the unnormalized laplacian matrix L of a weighted graph, defined
 * by:
 *
 * L(u,v) = dv - w(v,v)	if u = v
 *          -w(u,v)		if u and v are adjacent
 *			0			otherwise
 *
 * Assumes the graph represents each edge only once, ie. that the
 * adjacency lists are not bidirectional.
 *
 * @param graph graph to compute the laplacian from.
 * @return the laplacian matrix of the weighted graph.
 */
Mat_<double> laplacian(const WeightedGraph &graph);

/**
 * Computes the unnormalized laplacian matrix L of a weighted graph, defined
 * by:
 *
 * L(u,v) = dv - w(v,v)	if u = v
 *          -w(u,v)		if u and v are adjacent
 *			0			otherwise
 *
 * Assumes the graph represents each edge only once, ie. that the
 * adjacency lists are not bidirectional.
 *
 * @param graph graph to compute the laplacian from.
 * @return the laplacian matrix of the weighted graph.
 */
MatrixXd eigLaplacian(const WeightedGraph &graph);

/**
 * Computes the normalized laplacian NL matrix of a weighted graph, defined by:
 *
 * NL(u,v) = 1 - w(v,v)/dv			if u = v
 *			 -w(u,v)/sqrt(du*dv)	if u and v are adjacent
 *			0						otherwise
 * Assumes the graph represents each edge only once, ie. that the
 * adjacency lists are not bidirectional.
 *
 * @param graph graph to compute the laplacian from.
 * @return the normalized laplacian matrix of the weighted graph.
 */
Mat_<double> normalizedLaplacian(const WeightedGraph &graph);

/**
 * Computes the normalized laplacian NL matrix of a weighted graph, defined by:
 *
 * NL(u,v) = 1 - w(v,v)/dv			if u = v
 *			 -w(u,v)/sqrt(du*dv)	if u and v are adjacent
 *			 0						otherwise
 * Assumes the graph represents each edge only once, ie. that the
 * adjacency lists are not bidirectional.
 *
 * @param graph graph to compute the laplacian from.
 * @return the normalized laplacian matrix of the weighted graph.
 */
MatrixXd eigNormalizedLaplacian(const WeightedGraph &graph);

/**
 * Computes the adjacency matrix of a simple weighted graph. It is a
 * square, symmetrical matrix.
 *
 * @param graph graph to compute the adjacency matrix of.
 * @return adjacency matrix of the graph.
 */
MatrixXd eigAdjacencyMatrix(const WeightedGraph &graph);

/**
 * Same as laplacian, but returns a sparse matrix data structure.
 * Useful when dealing with graphs with a lot of vertices, but
 * relatively few edges (for instances grid graphs, nearest neighbor
 * graphs, large planar graphs).
 */
Eigen::SparseMatrix<double> sparseLaplacian(const WeightedGraph &graph, bool bidirectional, Eigen::VectorXd &degrees);

Eigen::SparseMatrix<double> _sparseLaplacian(const WeightedGraph &graph, bool bidirectional);

/**
 * Same as normalizedLaplacian, but returns a sparse matrix data structure.
 */
Eigen::SparseMatrix<double> normalizedSparseLaplacian(const WeightedGraph &graph, bool bidirectional, Eigen::VectorXd &degrees);

Eigen::SparseMatrix<double> _normalizedSparseLaplacian(const WeightedGraph &graph, bool bidirectional);

/**
 * Compute the normalized sparse laplacian as defined by (Shi and Malik 2000).
 * This matrix representation is NOT symmetric. Assumes the input graph is simple
 * eg. no loops or more than 1 edge between 2 given vertices.
 */
Eigen::SparseMatrix<double> randomWalkSparseLaplacian(const WeightedGraph &graph, bool bidirectional, Eigen::VectorXd &degrees);

/**
 * Computes the normalized laplacian of the graph in a upper triangular symmetric packed 
 * storage dense column major format intended for use with LAPACK. Expects a non bidirectional graph.
 *
 * @param graph graph to compute the normalized laplacian from.
 * @param L output array storage for the normalized laplacian. Memory management is up to the
 * caller, should be of size at least n(n + 1) / 2 where n is the number of vertices in the graph.
 */
void packedStorageNormalizedLaplacian(const WeightedGraph &graph, double *L);

/**
 * Solves a sparse, symmetric eigen system, computing a specific number of largest or smallest
 * eigenvalues and eigenvectors. Internally uses ARPACK's routines.
 *
 * @param L n by n square symmetric matrix to compute eigenvalues/eigenvectors from.
 * @param which string indicating which eigenvalues/eigenvectors to compute:
 * "LA" for the nev largest (algebraic) eigenvalues
 * "SA" for the nev smallest (algebraic) eigenvalues
 * "LM" for the nev largest (in magnitude) eigenvalues
 * "SM" for the nev smalles (in magnitude) eigenvalues
 * "BE" for nev eigenvalues, half from each end of the spectrum.
 * @param nev the number of eigenvalues/eigenvectors to compute.
 * @param evalues output nev-sized vector of eigenvalus of L.
 * @param evectors output n by nev sized matrix of eigenvectors of L.
 */
void symmetricSparseEigenSolver(const Eigen::SparseMatrix<double> &L, char *which, int nev, int maxIterations, Eigen::VectorXd &evalues, Eigen::MatrixXd &evectors);

/**
 * Solves a sparse, non-symmetric eigen system, computing a specific number of largest or smallest
 * eigenvalues and eigenvectors. Internally uses ARPACK's routines.
 *
 * @param L n by n square symmetric matrix to compute eigenvalues/eigenvectors from.
 * @param which string indicating which eigenvalues/eigenvectors to compute:
 * "LA" for the nev largest (algebraic) eigenvalues
 * "SA" for the nev smallest (algebraic) eigenvalues
 * "LM" for the nev largest (in magnitude) eigenvalues
 * "SM" for the nev smalles (in magnitude) eigenvalues
 * "BE" for nev eigenvalues, half from each end of the spectrum.
 * @param nev the number of eigenvalues/eigenvectors to compute.
 * @param evalues output nev-sized vector of eigenvalus of L.
 * @param evectors output n by nev sized matrix of eigenvectors of L.
 */
void nonSymmetricSparseEigenSolver(const Eigen::SparseMatrix<double> &L, char *which, int nev, int maxIterations, Eigen::VectorXd &evalues, Eigen::MatrixXd &evectors);

/**
 * Base functor class for user-defined matrix vector multiplication.
 */
class MatrixVectorMult {
public:
	/**
	 * Computes A * X = Y, where A is an m by n matrix, X and Y are
	 * n sized column vectors. It is up to the implementing subclass
	 * to define the data structure and storage of the matrix A,
	 * as sparse matrix private class member for instance.
	 */
	virtual void operator() (double *X, double *Y) = 0;
};

/**
 * Sparse matrix - dense vector multiplication as implemented by Eigen.
 */
class EigenMult : public MatrixVectorMult {
private:
	const Eigen::SparseMatrix<double> *L;

public:
	/**
	 * Initializes the functor with the sparse matrix to
	 * multiply vectors with.
	 */
	EigenMult(const Eigen::SparseMatrix<double> *L);
	void operator() (double *X, double *Y);
};

/**
 * Same as symmetricSparseEigenSolver, taking a user-defined matrix-vector multiplication
 * routine as parameter.
 *
 * @param order order of the matrix.
 * @param which string indicating which eigenvalues/eigenvectors to compute:
 * "LA" for the nev largest (algebraic) eigenvalues
 * "SA" for the nev smallest (algebraic) eigenvalues
 * "LM" for the nev largest (in magnitude) eigenvalues
 * "SM" for the nev smalles (in magnitude) eigenvalues
 * "BE" for nev eigenvalues, half from each end of the spectrum.
 * @param nev the number of eigenvalues/eigenvectors to compute.
 * @param evalues output nev-sized vector of eigenvalus of L.
 * @param evectors output n by nev sized matrix of eigenvectors of L.
 * @param mult user defined matrix vector multiplication routine.
 */
void symmetricSparseEigenSolver(int order, char *which, int nev, int maxIterations, Eigen::VectorXd &evalues, Eigen::MatrixXd &evectors, MatrixVectorMult &mult);

/**
 * Same as nonSymmetricSparseEigenSolver, taking a user-defined matrix-vector multiplication
 * routine as parameter.
 *
 * @param order order of the matrix.
 * @param which string indicating which eigenvalues/eigenvectors to compute:
 * "LA" for the nev largest (algebraic) eigenvalues
 * "SA" for the nev smallest (algebraic) eigenvalues
 * "LM" for the nev largest (in magnitude) eigenvalues
 * "SM" for the nev smalles (in magnitude) eigenvalues
 * "BE" for nev eigenvalues, half from each end of the spectrum.
 * @param nev the number of eigenvalues/eigenvectors to compute.
 * @param evalues output nev-sized vector of eigenvalus of L.
 * @param evectors output n by nev sized matrix of eigenvectors of L.
 * @param mult user defined matrix vector multiplication routine.
 */
void nonSymmetricSparseEigenSolver(int order, char *which, int nev, int maxIterations, Eigen::VectorXd &evalues, Eigen::MatrixXd &evectors, MatrixVectorMult &mult);

/**
 * Computes the eigengap of a vector of eigenvalues.
 *
 * @param eigenvalues vector of eigenvalues sorted in ascending order.
 * @return the index of the eigengap, that is k such that |ek - ek+1| is largest.
 */
int eigenGap(Eigen::VectorXd eigenvalues, double epsilon = 10E-4);