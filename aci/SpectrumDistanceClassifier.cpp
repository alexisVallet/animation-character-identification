#include "SpectrumDistanceClassifier.h"

SpectrumDistanceClassifier::SpectrumDistanceClassifier(TrainableStatModel *statModel, MatrixRepresentation representation, float mu) 
	: statModel(statModel), representation(representation), mu(mu)
{
	assert(mu > 0);
}

/*void SpectrumDistanceClassifier::train(vector<LabeledGraph<Mat> > trainingSamples, Mat &trainingClasses) {
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
}*/

static bool compareGraphSize(const WeightedGraph &g1, const WeightedGraph &g2) {
	return g1.numberOfVertices() < g2.numberOfVertices();
}

float SpectrumDistanceClassifier::leaveOneOutRecognitionRate(vector<WeightedGraph> samples, Mat_<int> &classes) {
	assert(samples.size() >= 2);
	assert(samples.size() == classes.rows);
	assert(classes.cols == 1);
	// first determine which graph has the largest number of vertices
	int maxNumberOfVertices = max_element(samples.begin(), samples.end(), compareGraphSize)->numberOfVertices();

	// then compute the eigenvalues of the matrix representation of each samples
	Mat_<float> spectra = Mat_<float>::zeros(samples.size(), maxNumberOfVertices);
	vector<Mat_<double> > reps;
	reps.reserve(samples.size());

	for (int i = 0; i < samples.size(); i++) {
		Mat_<double> matRep = this->representation(samples[i]);
		reps.push_back(matRep);
		Mat_<double> largerMatRep = Mat_<double>::zeros(maxNumberOfVertices, maxNumberOfVertices);
		matRep.copyTo(largerMatRep.rowRange(0,matRep.rows).colRange(0,matRep.cols));

		Mat_<double> eigenvalues;

		eigen(largerMatRep, eigenvalues);

		Mat_<float> evf = Mat_<float>(eigenvalues.t());

		evf.copyTo(spectra.row(i));
	}

	return this->statModel->leaveOneOutCrossValidation(spectra, classes);
}
