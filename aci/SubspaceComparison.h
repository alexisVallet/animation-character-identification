#pragma once

#include <iostream>
#include <Eigen/Dense>

#include "Utils.hpp"

using namespace Eigen;
using namespace std;

/**
 * Given two m by p matrices A and B, this function finds the orthogonal matrix
 * Q such that ||A - BQ|| is minimized, where ||.|| denotes the Frobenius norm.
 * Algorithm from Matrix Computations, 1996, section 12.4.1.
 *
 * @param A a m by p matrix.
 * @param B a m by p matrix.
 * @param Q output orthogonal matrix minimizing ||A - BQ|| .
 */
void subspacesRotation(const MatrixXd &A, const MatrixXd &B, MatrixXd &Q);

/**
 * Computes the intersection of the null spaces of two matrices A and B as
 * a basis for the intersection.
 * Algorithm from Matrix Computations, 1996, section 12.4.2.
 *
 * @param A m by n matrix.
 * @param B p by n matrix.
 * @param Y output matrix containing a basis as column vector for the intersection
 * of null(A) and null(B). May be empty if the intersection is trivial (ie only
 * the origin).
 */
void nullSpacesIntersection(const MatrixXd &A, const MatrixXd &B, MatrixXd &Y);

/**
 * Computes the principal angles and vectors betweem two subspaces represented
 * by bases A and B, where columns correspond to vectors.
 * Algorithm from Matrix Computations, 1996, section 12.4.3.
 *
 * @param A m by p matrix.
 * @param B m by q matrix.
 * @param U output matrix containing principal vectors u1 ... uq as columns.
 * @param V output matrix containing principal vectors v1 ... vq as columns.
 * @param cosines cosines of the canonical angles between range(A) and range(B),
 * sorted in decreasing order.
 */
void canonicalAngles(const MatrixXd &A, const MatrixXd &B, MatrixXd &U, MatrixXd &V, VectorXd &cosines);

/**
 * Computes the intersection of two subspaces, each represented by an orthonormal
 * basis as columns.
 *
 * @param A m by p matrix representing the first subspace.
 * @param B m by q matrix representing the second subspace.
 * @param C matrix containing an orthonormal basis for the intersection of
 * range(A) and range(B).
 */
void subspacesIntersection(const MatrixXd &A, const MatrixXd &B, MatrixXd &C);

/**
 * Computes the distance between 2 equidimensional subspaces represented each by
 * an orthonormal basis. Distance is defined as in Matrix Computations, 1996.
 *
 * @param A n by m matrix containing an orthonormal basis as columns.
 * @param B n by m matrix containing an orthonormal basis as columns.
 */
double subspaceDistance(const MatrixXd &A, const MatrixXd &B);
