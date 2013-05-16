#include "SpectrumDistanceClassifier.h"

SpectrumDistanceClassifier::SpectrumDistanceClassifier(TrainableStatModel *statModel, MatrixRepresentation representation, int k) 
	: statModel(statModel), representation(representation), k(k)
{
	assert(k > 0);
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

float SpectrumDistanceClassifier::leaveOneOutRecognitionRate(vector<WeightedGraph> samples, const Mat_<int> &classes) {
	assert(samples.size() >= 2);
	assert(samples.size() == classes.rows);
	assert(classes.cols == 1);

	// then compute the first k non 0 eigenvalues of the matrix representation of each samples
	Mat_<float> spectra = Mat_<float>::zeros(samples.size(), this->k);
	vector<Mat_<double> > reps;
	reps.reserve(samples.size());

	for (int i = 0; i < samples.size(); i++) {
		Mat_<double> matRep = this->representation(samples[i]);
		reps.push_back(matRep);

		Mat_<double> eigenvalues;

		eigen(matRep, eigenvalues);

		Mat_<double> ascEigenvalues;

		flip(eigenvalues, ascEigenvalues, 0);

		Mat_<float> evf = Mat_<float>(ascEigenvalues.t());

		int nbToCopy = min(k + 2, evf.cols);

		evf.colRange(2,nbToCopy).copyTo(spectra.row(i).colRange(0, nbToCopy - 2));

		//cout<<spectra.row(i)<<endl;
	}

	return this->statModel->leaveOneOutCrossValidation(spectra, classes);
}
