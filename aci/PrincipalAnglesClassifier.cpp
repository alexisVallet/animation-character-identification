#include "PrincipalAnglesClassifier.h"

#define DEBUG_PRINCIPALANGLESCLASSIFIER true

PrincipalAnglesClassifier::PrincipalAnglesClassifier(SimilarityMeasure similarity, DenseRepresentation laplacian, bool symmetric, bool smallestEigenvalues) 
	: similarity(similarity), laplacian(laplacian), symmetric(symmetric), smallestEigenvalues(smallestEigenvalues)
{

}

MatrixXd PrincipalAnglesClassifier::graphSubspace(const WeightedGraph &graph) {
	// compute the laplacian matrix of the sample as well as
	// its eigenvalues and eigenvectors
	MatrixXd laplacianMatrix = this->laplacian(graph);
	VectorXd eigenvalues;
	MatrixXd eigenvectors;	

	// distinguishes the cases of symmetric matrices for better performance.
	// Some Laplacian matrices, e.g. the random walk Laplacian, are not
	// symmetric.
	if (this->symmetric) {
		SelfAdjointEigenSolver<MatrixXd> solver(laplacianMatrix);

		eigenvalues = solver.eigenvalues();
		eigenvectors = solver.eigenvectors();
	} else {
		EigenSolver<MatrixXd> solver(laplacianMatrix);

		eigenvalues = solver.eigenvalues().real();
		eigenvectors = solver.eigenvectors().real();
	}

	// compute the eigengap k, and keep only the k first eigenvectors.
	int k = eigenGap(eigenvalues) + 1;

	if (DEBUG_PRINCIPALANGLESCLASSIFIER) {
		cout<<"L:"<<endl<<laplacianMatrix<<endl;
		cout<<"eigenvalues:"<<endl<<eigenvalues<<endl;
		cout<<"eigengap = "<<k<<endl;
	}

	// distinguish the case where we want smallest eigenvalue eigenvectors from
	// the case of the largest eigenvalue eigenvectors. The latter only happens
	// when we use the adjacency matrix directly.
	if (this->smallestEigenvalues) {
		return eigenvectors.block(0, 0, eigenvectors.rows(), k);
	} else {
		return eigenvectors.block(0, eigenvectors.cols() - k, eigenvectors.rows(), k);
	}
}

void PrincipalAnglesClassifier::train(const vector<pair<WeightedGraph,int> > &trainingSet) {
	this->subspaces.reserve(trainingSet.size());

	for (int i = 0; i < trainingSet.size(); i++) {
		// compute and register the subspace into the training data.
		this->subspaces.push_back(
			pair<MatrixXd, int>(
				this->graphSubspace(trainingSet[i].first), 
				trainingSet[i].second));
	}
}

double PrincipalAnglesClassifier::similarityFunction(const MatrixXd &s1, const MatrixXd &s2) {
	// compute canonical angles between the two subspaces
	VectorXd cosines;
	MatrixXd U, V;

	canonicalAngles(s1, s2, U, V, cosines);

	if (DEBUG_PRINCIPALANGLESCLASSIFIER) {
		cout<<"s1:"<<endl<<s1<<endl;
		cout<<"s2:"<<endl<<s2<<endl;
		cout<<"canonical angles cosines:"<<endl<<cosines<<endl;
	}

	// Depending on the desired similarity, either
	// return the average cosine or the maximum cosine.
	// Since the canonical angles are in the [0;pi/2] range, cosines closest to 1
	// correspond to more similar dimensions, hence the use of the maximum and
	// average cosine directly instead of angles. Although the relationship between
	// cosine and angle is not linear, so this may not be so good.
	switch (this->similarity) {
	case PrincipalAnglesClassifier::AVERAGE_ANGLE :
		return cosines.mean();
		break;
	case PrincipalAnglesClassifier::SMALLEST_ANGLE :
		return cosines.maxCoeff();
		break;
	};
}

static MatrixXd zeroPadding(const MatrixXd &toPad, int newRows, int newCols) {
	assert(newRows >= toPad.rows() && newCols >= toPad.cols());
	MatrixXd padded = MatrixXd::Zero(newRows, newCols);

	padded.block(0,0,toPad.rows(),toPad.cols()) = toPad;

	return padded;
}

static bool compareNbRows(const pair<MatrixXd,int> t1, const pair<MatrixXd,int> t2) {
	return t1.first.rows() < t2.first.rows();
}

int PrincipalAnglesClassifier::predict(const WeightedGraph &testSample) {
	// compute the test graph subspace, pad it appropriately with zeros
	MatrixXd testSubspace = this->graphSubspace(testSample);
	// compute the dimension of the embedding space by taking the largest subspace
	// dimension. The dimension of the subspace is the number of rows in the basis
	// matrix, which in turns corresponds to the number of vertices of the graph.
	int dimension = 
		max(
			testSample.numberOfVertices(), 
			max_element(
				this->subspaces.begin(), 
				this->subspaces.end(), 
				compareNbRows)->first.rows());
	MatrixXd paddedTest = zeroPadding(testSubspace, dimension, testSubspace.cols());

	// compute similarities between the test sample and all training samples,
	// keeps the max.
	double maxSimilarity = 0;
	int maxIdx = 0;

	for (int i = 0; i < this->subspaces.size(); i++) {
		MatrixXd paddedTraining = 
			zeroPadding(
				this->subspaces[i].first, 
				dimension, 
				this->subspaces[i].first.cols());
		double currentSimilarity = this->similarityFunction(paddedTest, paddedTraining);
		if (currentSimilarity > maxSimilarity) {
			maxSimilarity = currentSimilarity;
			maxIdx = i;
		}
	}

	return this->subspaces[maxIdx].second;
}
