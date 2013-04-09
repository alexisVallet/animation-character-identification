#include "SpectrumDistanceClassifier.h"

SpectrumDistanceClassifier::SpectrumDistanceClassifier(MatKernel kernel, TrainableStatModel *statModel, MatrixRepresentation representation, int maxNumberOfVertices) 
	: kernel(kernel), statModel(statModel), representation(representation), maxNumberOfVertices(maxNumberOfVertices)
{
	assert(maxNumberofVertices > 0);
}

void SpectrumDistanceClassifier::train(vector<LabeledGraph<Mat> > trainingSamples, Mat &trainingClasses) {
	Mat samplesMatrix(trainingSamples.size(), this->maxNumberOfVertices, CV_32F);

	assert(traininSamples.size() == trainingClasses.rows);

	for (int i = 0; i < trainingSamples.size(); i++) {
		WeightedGraph weighted = weighEdgesByKernel<Mat>(this->kernel, trainingSamples[i]);
		Mat_<double> matrix = this->representation(weighted);
		Mat eigenvalues;

		eigen(matrix, eigenvalues);

		samplesMatrix.row(i) = eigenvalues;
	}

	this->statModel->train(samplesMatrix, trainingClasses);
}

int SpectrumDistanceClassifier::predict(LabeledGraph<Mat> testSample) {
	WeightedGraph weighted = weighEdgesByKernel<Mat>(this->kernel, testSample);
	Mat_<double> matrix = this->representation(weighted);
	Mat eigenvalues;

	eigen(matrix, eigenvalues);

	return this->statModel->predict(eigenvalues);
}
