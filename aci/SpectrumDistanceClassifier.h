/** @file */
#pragma once

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cfloat>

#include "TrainableStatModel.h"
#include "Kernels.h"
#include "GraphSpectra.h"

using namespace cv;
using namespace std;

/**
 * Classify graphs by the spectrum of their respective Laplacian matrices, with
 * edges weighted by a kernel function between color histograms of image segments.
 * Can work for supervised classification tasks as well as unsupervised classification
 * (clustering).
 */
class SpectrumDistanceClassifier {
private:
	TrainableStatModel *statModel;
	MatrixRepresentation representation;
	int k;

public:
	/**
	 * Initialize the classifier with a specific classification model (KNN, Bayes, ...)
	 * a graph matrix representation (combinatorial Laplacian, normalized Laplacian, ...)
	 * and a number of eigenvalues to take into account for classification.
	 *
	 * @param statModel statistical model to classify spectra with.
	 * @param representation the matrix representation to use for graphs.
	 * @param k number of non zero smallest eigenvalues to use for classification.
	 */
	SpectrumDistanceClassifier(TrainableStatModel *statModel, MatrixRepresentation representation, int k);

	/**
	 * Trains the classifier with specific training samples.
	 *
	 * @param trainingSamples graphs to train the classifier with.
	 * @param trainingClasses class for each of the training samples.
	 */
	//void train(vector<LabeledGraph<Mat> > trainingSamples, Mat &trainingClasses);

	/**
	 * Predicts the class a specific test sample belongs to using training data. Must call train
	 * before calling this method.
	 *
	 * @param testSample the graph to classify.
	 * @return the class the sample is predicted to belong to.
	 */
	//int predict(LabeledGraph<Mat> testSample);

	/**
	 * Computes the leave one out recognition rate for a given sample. It means that to classify
	 * a sample, the classifier is trained with all the other samples. The recognition rate is
	 * then the number of correctly classified samples divided by the total number of samples.
	 *
	 * @param samples the samples to compute the recognition rate from.
	 * @param classes the class each sample belongs to.
	 * @return a recognition rate within the real interval [0; 1].
	 */
	float leaveOneOutRecognitionRate(vector<WeightedGraph> samples, const Mat_<int> &classes);


};
