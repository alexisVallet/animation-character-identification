#pragma once

#include <opencv2\opencv.hpp>

#include "TrainableStatModel.h"
#include "Kernels.h"
#include "GraphSpectra.h"

using namespace cv;
using namespace std;

/**
 * Classify graphs by the spectrum of their respective Laplacian matrices, with
 * edges weighted by a kernel function between color histograms of image segments.
 * It is a supervised classifier.
 */
class SpectrumDistanceClassifier {
private:
	MatKernel kernel;
	TrainableStatModel *statModel;
	MatrixRepresentation representation;
	int maxNumberOfVertices;

public:
	/**
	 * Initializes the classifier with a specific kernel function to compute weights,
	 * and a specific statistical model to classify Laplacian spectra with. Requires
	 * a maximum number of vertices for graphs to classify to be specified.
	 *
	 * @param kernel kernel function to compute weights from.
	 * @param statModel statistical model to classify Laplacian spectra with.
	 * @param maxNumberOfVertices an upper bound on the number of vertices.
	 */
	SpectrumDistanceClassifier(MatKernel kernel, TrainableStatModel *statModel, MatrixRepresentation representation, int maxNumberOfVertices);

	/**
	 * Trains the classifier with specific training samples.
	 *
	 * @param trainingSamples graphs to train the classifier with.
	 * @param trainingClasses class for each of the training samples.
	 */
	void train(vector<LabeledGraph<Mat> > trainingSamples, Mat &trainingClasses);

	/**
	 * Predicts the class a specific test sample belongs to using training data. Must call train
	 * before calling this method.
	 *
	 * @param testSample the graph to classify.
	 * @return the class the sample is predicted to belong to.
	 */
	int predict(LabeledGraph<Mat> testSample);
};
