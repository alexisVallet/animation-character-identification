#include "PrincipalAnglesClassifier.h"

#define TEST_PRINCIPALANGLESCLASSIFIER true

PrincipalAnglesClassifier::PrincipalAnglesClassifier(TrainableStatModel* statModel, DenseRepresentation representation, bool bidirectional, bool symmetric, int vsize)
	: statModel(statModel), representation(representation), bidirectional(bidirectional), symmetric(symmetric), vsize(vsize)
{

}

void PrincipalAnglesClassifier::graphPrincipalAngles(const WeightedGraph &graph, const MatrixXd &id, VectorXd &cosines) {
		// compute laplacian matrix
		MatrixXd laplacian = this->representation(graph);

		// compute eigenvalues and eigenvectors of the laplacian
		VectorXd evalues;
		MatrixXd evectors;
		int nev = min(graph.numberOfVertices(), this->vsize);

		if (this->symmetric) {
			SelfAdjointEigenSolver<MatrixXd> solver(laplacian);

			evalues = solver.eigenvalues();
			evectors = solver.eigenvectors();
		} else {
			EigenSolver<MatrixXd> solver(laplacian);

			// as general eigen solving may yield complex values, here we
			// assume that if the Laplacian is not symmetric - for instance
			// with the random walk Laplacian - then it still has real eigenvalues.
			evalues = solver.eigenvalues().real();
			evectors = solver.eigenvectors().real();
		}

		

		// compute the eigengap
		int k = eigenGap(evalues);
		

		// compute principal angles cosines with appropriately padded matrices
		int n = graph.numberOfVertices();
		MatrixXd U, V;

		canonicalAngles(id.block(0,0,n,n), evectors.block(0,0,n,k + 1), U, V, cosines);
		
		if (TEST_PRINCIPALANGLESCLASSIFIER) {
			/*cout<<"L = "<<endl<<laplacian<<endl;
			cout<<"evalues = "<<evalues<<endl;
			cout<<"evectors = "<<endl<<evectors<<endl;
			cout<<"eigengap at "<<k<<endl;*/
			cout<<"canonical angles = "<<cosines.transpose()<<endl;
		}
}

void PrincipalAnglesClassifier::train(const vector<pair<WeightedGraph, int> > &trainingSet) {
	Mat_<double> trainData = Mat_<double>::zeros(trainingSet.size(), this->vsize);
	Mat_<int> expectedResponses(trainingSet.size(), 1);
	MatrixXd canonicalBasis = MatrixXd::Identity(this->vsize, this->vsize);

	for (int i = 0; i < (int)trainingSet.size(); i++) {
		VectorXd cosines;

		this->graphPrincipalAngles(trainingSet[i].first, canonicalBasis, cosines);
		
		// copy canonical angle cosines into training matrix padded with zeros
		for (int j = 0; j < (int)cosines.size(); j++) {
			trainData(i,j) = cosines(j);
		}
		expectedResponses(i,0) = trainingSet[i].second;
	}

	this->statModel->train(trainData, expectedResponses, Mat());
}

int PrincipalAnglesClassifier::predict(WeightedGraph &testSample) {
	VectorXd cosines;
	MatrixXd canonicalBasis = MatrixXd::Identity(this->vsize, this->vsize);

	this->graphPrincipalAngles(testSample, canonicalBasis, cosines);

	Mat_<double> paddedCosines = Mat_<double>::zeros(1, this->vsize);

	for (int i = 0; i < (int)cosines.size(); i++) {
		paddedCosines(0,i) = cosines(i);
	}

	return this->statModel->predict(paddedCosines);
}

float PrincipalAnglesClassifier::leaveOneOutRecognitionRate(vector<pair<WeightedGraph,int> > samples) {
	Mat_<float> sampleVectors = Mat_<float>::zeros(samples.size(), this->vsize);
	Mat_<int> classes(samples.size(), 1);
	MatrixXd canonicalBasis = MatrixXd::Identity(this->vsize, this->vsize);
	
	for (int i = 0; i < (int)samples.size(); i++) {
		VectorXd cosines;

		this->graphPrincipalAngles(samples[i].first, canonicalBasis, cosines);

		for (int j = 0; j < cosines.size(); j++) {
			sampleVectors(i,j) = cosines(j);
		}

		classes(i,0) = samples[i].second;
	}

	return this->statModel->leaveOneOutCrossValidation(sampleVectors, classes);
}
