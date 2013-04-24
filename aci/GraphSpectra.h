#pragma once

#include <Eigen\Sparse>
#include <opencv2\opencv.hpp>
#include <complex>
#include <limits>

#include "WeightedGraph.hpp"
#include "Utils.hpp"

using namespace std;
using namespace cv;

typedef Mat_<double> (*MatrixRepresentation)(const WeightedGraph&);

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
 * Same as laplacian, but returns a sparse matrix data structure.
 * Useful when dealing with graphs with a lot of vertices, but
 * relatively few edges (for instances grid graphs, nearest neighbor
 * graphs, large planar graphs).
 */
Eigen::SparseMatrix<double> sparseLaplacian(const WeightedGraph &graph, bool bidirectional, Eigen::VectorXd &degrees);

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
 * Same as normalizedLaplacian, but returns a sparse matrix data structure. Assumes
 * a bidirectional graph representation without loops.
 */
Eigen::SparseMatrix<double> normalizedSparseLaplacian(const WeightedGraph &graph, Eigen::VectorXd &degrees = Eigen::VectorXd(1));

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
void symmetricSparseEigenSolver(const Eigen::SparseMatrix<double> &L, char *which, int nev, Eigen::VectorXd &evalues, Eigen::MatrixXd &evectors);