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

#define DEFAULT_MU 0.001

/**
 * Classify graphs by the spectrum of their respective Laplacian matrices, with
 * edges weighted by a kernel function between color histograms of image segments.
 * It is a supervised classifier.
 */
class SpectrumDistanceClassifier {
private:
	TrainableStatModel *statModel;
	MatrixRepresentation representation;
	float lambda;
	float mu;

public:
	/**
	 * Initializes the classifier with a specific kernel function to compute weights,
	 * and a specific statistical model to classify graph spectra with. Eigenvalues
	 * are scaled with an inverse exponential to favor smaller eigenvalues.
	 *
	 * @param kernel kernel function to compute weights from.
	 * @param statModel statistical model to classify spectra with.
	 * @param representation the matrix representation to use for graphs.
	 * @param mu positive factor to the exponential favorizing smaller eigenvalues.
	 */
	SpectrumDistanceClassifier(TrainableStatModel *statModel, MatrixRepresentation representation, float mu = DEFAULT_MU);

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
	float leaveOneOutRecognitionRate(vector<WeightedGraph> samples, Mat_<int> &classes);
};
