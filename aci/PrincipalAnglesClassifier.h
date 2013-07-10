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
 * CLassifies graphs according to the principal angles of the space spanned
 * by the first eigenvectors of its laplacian against the canonical basis.
 * Note that the number of eigenvectors taken per graph is not constant, and
 * is determined through the eigengap heuristic.
 */
class PrincipalAnglesClassifier {
private:
	TrainableStatModel *statModel;
	const SparseRepresentation representation;
	const bool bidirectional;
	const bool symmetric;
	const int vsize;

	void graphPrincipalAngles(const WeightedGraph &graph, const MatrixXd &id, VectorXd &cosines);

public:
	/**
	 * Initializes the classifier with a specific statistical model (e.g. SVM,
	 * KNN, ANN...) and some sparse matrix representation for the graphs which may
	 * or may not be symmetric, as specified by a flag.
	 *
	 * @param statModel statistical model to classify principal angles vectors with.
	 * @param representation sparse matrix representation for some Laplacian matrix.
	 * @param bidirectional true iff the matrix representation requires bidirectional
	 * graph data structure.
	 * @param symmetric true iff the matrix representation is symmetric.
	 * @param vsize size of the vectors to classify.
	 */
	PrincipalAnglesClassifier(TrainableStatModel *statModel, SparseRepresentation representation, bool bidirectional, bool symmetric, int vsize);

	/**
	 * Trains the classifier using a specific training set.
	 *
	 * @param trainingSet set of (sample,class) pairs to train the classifier with.
	 */
	void train(const vector<pair<WeightedGraph, int> > &trainingSet);

	/**
	 * Predicts the class label of a test sample after training.
	 *
	 * @param testSample sample to predict the class label of.
	 */
	int predict(WeightedGraph &testSample);

	/**
	 * Compute a recognition rate using leave one out cross validation from
	 * a set of labeled samples.
	 *
	 * @param samples set of labeled samples to compute the recognition rate
	 * from.
	 * @return a recognition rate for the samples.
	 */
	float leaveOneOutRecognitionRate(vector<pair<WeightedGraph,int> > samples);
};
