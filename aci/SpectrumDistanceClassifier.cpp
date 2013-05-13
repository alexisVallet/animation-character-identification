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

float SpectrumDistanceClassifier::leaveOneOutRecognitionRate(vector<WeightedGraph> samples, Mat &classes) {
	assert(samples.size() == classes.rows);
	// first determine the maximum number of vertices
	int maxNumberOfVertices = (*max_element(samples.begin(), samples.end(), compareGraphSize)).numberOfVertices();
	cout<<"max nb vertices: "<<maxNumberOfVertices<<endl;

	// compute the spectrum of each graph
	Mat spectra = Mat::zeros(samples.size(), maxNumberOfVertices, CV_32F);

	for (int i = 0; i < (int)samples.size(); i++) {
		cout<<"computing representation"<<endl;
		Mat_<double> matrix = this->representation(samples[i]);
		cout<<"padding representation"<<endl;
		Mat_<double> largerMatrix = Mat_<double>::zeros(maxNumberOfVertices, maxNumberOfVertices);

		matrix.copyTo(largerMatrix.rowRange(0, matrix.rows).colRange(0, matrix.cols));

		Mat eigenvalues;

		eigen(largerMatrix, eigenvalues);

		eigenvalues = eigenvalues.t();

		// scales so the smallest non zero eigenvalues matter more
		/*for (int j = 0; j < eigenvalues.cols; j++) {
			if (eigenvalues.at<double>(0,j) > 10E-8) {
				eigenvalues.at<double>(0,j) = exp(-this->mu * eigenvalues.at<double>(0,j));
			}
		}*/

		eigenvalues.copyTo(spectra.row(i));
	}

	// for each sample, trains the classifier with the other samples then classify.
	Mat otherSamples = Mat::zeros(spectra.rows - 1, maxNumberOfVertices, CV_32F);
	Mat otherClasses = Mat::zeros(spectra.rows - 1, 1, CV_32S);
	spectra.rowRange(1, spectra.rows).copyTo(otherSamples);
	classes.rowRange(1, spectra.rows).copyTo(otherClasses);
	int totalCorrectResults = 0;

	for (int i = 0; i < spectra.rows; i++) {
		this->statModel->clear();
		this->statModel->train(otherSamples, otherClasses);
		int result = (int)floor(this->statModel->predict(spectra.row(i)));

		cout<<"actual = "<<result<<", expected = "<<classes.at<int>(i,0)<<endl;

		cout<<"sample = "<<spectra.row(i)<<endl;
		float closestDistance = FLT_MAX;
		int best = 0;

		for (int j = 0; j < spectra.rows; j++) {
			if (i != j) {
				float distance = (float)norm(spectra.row(j), spectra.row(i));

				if (distance < closestDistance) {
					closestDistance = distance;
					best = j;
				}
			}
		}

		cout<<"closest at "<<best<<" distance "<<closestDistance<<" = "<<spectra.row(best)<<endl;

		if (result == classes.at<int>(i,0)) {
			totalCorrectResults++;
		}

		if (i < spectra.rows - 1) {
			spectra.row(i).copyTo(otherSamples.row(i));
			classes.row(i).copyTo(otherClasses.row(i));
		}
	}

	return ((float)totalCorrectResults)/((float)samples.size());
}
