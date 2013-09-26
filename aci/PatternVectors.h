/** @file */
#pragma once

#include <Eigen/Dense>
#include <vector>

#include "WeightedGraph.hpp"
#include "GraphSpectra.h"

using namespace Eigen;
using namespace std;

/**
 * Evaluates the power symmetric polynomials, or the power sum, on the
 * coefficients of a vector.
 *
 * @param inputs vector whose coefficients should be used as inputs to the
 * polynomials.
 * @return the result of the n first power sums for the given inputs, where n
 * is the number of coefficients of the inputs.
 */
VectorXd evaluatePowerSymmetricPolynomials(const VectorXd &inputs);

/**
 * Evaluates the n first elementary symmetric polynomials on the coefficients
 * of an input vector.
 *
 * @param inputs vector whose coefficients should be used as inputs to the
 * polynomials.
 * @param the result of the n first elementary symmetric polynomials for the 
 * given inputs, where n is the number of coefficients of the inputs.
 */
VectorXd evaluateSymmetricPolynomials(const VectorXd &inputs);

/**
 * Computes pattern vectors as described by Wilson, Luo, Hancock for a set
 * of graphs of varying size.
 *
 * @param graphs graphs to compute pattern vectors for.
 * @param k number of eigenvectors to consider for each graph. Must be
 * smaller than the number of vertices of the smallest input graphs.
 * @return pattern vector for each graph. Each vector has n*k components,
 * where n is the size of the largest input graph.
 */
vector<VectorXd> patternVectors(vector<WeightedGraph> &graphs, int k, int maxGraphSize);
