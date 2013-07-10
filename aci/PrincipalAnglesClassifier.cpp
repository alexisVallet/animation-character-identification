#include "PrincipalAnglesClassifier.h"

PrincipalAnglesClassifier::PrincipalAnglesClassifier(TrainableStatModel* statModel, SparseRepresentation representation, bool bidirectional, bool symmetric, int vsize)
	: statModel(statModel), representation(representation), bidirectional(bidirectional), symmetric(symmetric), vsize(vsize)
{

}

void PrincipalAnglesClassifier::graphPrincipalAngles(const WeightedGraph &graph, const MatrixXd &id, VectorXd &cosines) {
		// compute laplacian matrix
		SparseMatrix<double> laplacian = this->representation(graph, this->bidirectional);

		// add padding for the laplacian so arpack can compute all the eigenvalues
		SparseMatrix<double> paddedLaplacian(graph.numberOfVertices() + 1, graph.numberOfVertices() +1);
		vector<Triplet<double> > tripletList;
		tripletList.reserve(laplacian.nonZeros());

		for (int i = 0; i < laplacian.outerSize(); ++i) {
			for (SparseMatrix<double>::InnerIterator it(laplacian,i); it; ++it) {
				tripletList.push_back(Triplet<double>(it.row(), it.col(), it.value()));
			}
		}
		paddedLaplacian.setFromTriplets(tripletList.begin(), tripletList.end());
		cout<<"L = "<<endl<<paddedLaplacian<<endl;

		// compute eigenvalues and eigenvectors of the laplacian
		VectorXd evalues;
		MatrixXd evectors;
		int nev = min(graph.numberOfVertices(), this->vsize);

		if (this->symmetric) {
			symmetricSparseEigenSolver(paddedLaplacian, "SM", nev, laplacian.rows(), evalues, evectors);
		} else {
			nonSymmetricSparseEigenSolver(paddedLaplacian, "SM", nev, laplacian.rows(), evalues, evectors);
		}

		cout<<"evalues = "<<evalues<<endl;
		cout<<"evectors = "<<endl<<evectors<<endl;

		// compute the eigengap
		int k = eigenGap(evalues);
		cout<<"eigengap at "<<k<<endl;

		// compute principal angles cosines with appropriately padded matrices
		int n = graph.numberOfVertices();
		MatrixXd U, V;

		canonicalAngles(id.block(0,0,n,n), evectors.block(0,0,n,k + 1), U, V, cosines);
		cout<<"canonical angles = "<<cosines<<endl;
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
