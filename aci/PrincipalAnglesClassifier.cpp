#include "PrincipalAnglesClassifier.h"

PrincipalAnglesClassifier::PrincipalAnglesClassifier(TrainableStatModel* statModel, SparseRepresentation representation, bool bidirectional, bool symmetric, int vsize)
	: statModel(statModel), representation(representation), bidirectional(bidirectional), symmetric(symmetric), vsize(vsize)
{

}

void PrincipalAnglesClassifier::graphPrincipalAngles(const WeightedGraph &graph, const MatrixXd &id, VectorXd &cosines) {
		// compute laplacian matrix
		SparseMatrix<double> laplacian = this->representation(graph, this->bidirectional);

		// compute eigenvalues and eigenvectors of the laplacian
		VectorXd evalues;
		MatrixXd evectors;
		int nev = min(graph.numberOfVertices(), this->vsize);

		if (this->symmetric) {
			symmetricSparseEigenSolver(laplacian, "SM", nev, laplacian.rows(), evalues, evectors);
		} else {
			nonSymmetricSparseEigenSolver(laplacian, "SM", nev, laplacian.rows(), evalues, evectors);
		}

		// compute the eigengap
		int k = eigenGap(evalues);

		// compute principal angles cosines with appropriately padded matrices
		int n = graph.numberOfVertices();
		MatrixXd U, V;

		canonicalAngles(id.block(0,0,n,n), evectors.block(0,0,n,k), U, V, cosines);
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
