#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Sparse>

#include "TrainableStatModel.h"
#include "Kernels.h"
#include "GraphSpectra.h"
#include "SubspaceComparison.h"

using namespace cv;
using namespace std;
using namespace Eigen;

/**
 * Classifies weighted graphs using the principal angles as a measure of
 * similarity between the subspaces spanned by their smallest Laplacian
 * eigenvectors. Determines the number of eigenvectors to take using the
 * eigengap heuristic.
 */
class PrincipalAnglesClassifier {
public:
	enum SimilarityMeasure { AVERAGE_ANGLE, SMALLEST_ANGLE };

private:
	vector<pair<MatrixXd,int> > subspaces;
	SimilarityMeasure similarity;
	DenseRepresentation laplacian;
	bool symmetric;
	bool smallestEigenvalues;

	double similarityFunction(const MatrixXd &s1, const MatrixXd &s2);
	MatrixXd graphSubspace(const WeightedGraph &graph);
public:
	/**
	 * Initializes the classifier, specifying the type of similarity measure
	 * to use between subspaces and the Laplacian matrix to use.
	 *
	 * @param similarity type of similarity measure to use between subspaces, 
	 * between two cases:
	 * - AVERAGE : similarity is computed using the average canonical angle.
	 * - SMALLEST : similarity is computed using only the smallest canonical angle.
	 * @param laplacian type of laplacian matrix to use.
	 * @param symmetric true iff the chosen laplacian matrix is symmetric.
	 * @param smallestEigenvalues true if we consider the k smallest eigenvalues.
	 * of the laplacian representation, false if we consider the k largest.
	 */
	PrincipalAnglesClassifier(SimilarityMeasure similarity, DenseRepresentation laplacian, bool symmetric, bool smallestEigenvalues = true);

	/**
	 * Trains the classifier using some training samples, associated with
	 * an integer class label.
	 * Computes and stores the subspaces spanned by the smallest eigenvectors
	 * of each graph's Laplacian, using the eigengap heuristic to determine
	 * eigenvectors to take into account.
	 *
	 * @param trainingSet vector containing pairs of samples with their associated
	 * integer class label.
	 */
	void train(const vector<pair<WeightedGraph,int> > &trainingSet);

	/**
	 * Predicts the class label of a test sample from the last training data.
	 * Must have called the train method before this method.
	 * Returns the label of the most similar training sample, where similarity
	 * is measured either through average canonical angle between subspaces or
	 * smallest canonical angle.
	 *
	 * @param testSample test graph to predict the class label of.
	 * @return the predicted class label of testSample.
	 */
	int predict(const WeightedGraph &testSample);
};
