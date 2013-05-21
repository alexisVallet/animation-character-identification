#include "SpectralClustering.h"

void spectralClustering(const WeightedGraph &simGraph, SparseRepresentation matRep, int k, VectorXi &classLabels, bool normalize, bool symmetric) {
	assert(k >= 1);
	
	// compute the matrix representation of the graph
	SparseMatrix<double> rep = matRep(simGraph);

	cout<<rep<<endl;

	// compute its k smallest eigenvectors
	VectorXd eigenvalues;
	MatrixXd eigenvectors;

	symmetricSparseEigenSolver(rep, "SM", k, simGraph.numberOfVertices(), eigenvalues, eigenvectors);

	cout<<eigenvectors<<endl<<endl;;

	// normalize embedding coordinates if necessary
	if (normalize) {
		for (int i = 0; i < simGraph.numberOfVertices(); i++) {
			eigenvectors.row(i).normalize();
		}
	}

	cout<<eigenvectors<<endl;

	// cluster the embedding using K-means
	MatrixXf fEigenvectors = eigenvectors.cast<float>();
	Mat_<float> embeddings(simGraph.numberOfVertices(), k);

	for (int i = 0; i < simGraph.numberOfVertices(); i++) {
		for (int j = 0; j < k; j++) {
			embeddings(i,j) = fEigenvectors(i,j);
		}
	}

	Mat labels;

	kmeans(embeddings, k, labels, TermCriteria(), 1, KMEANS_RANDOM_CENTERS);

	// compute the corresponding clusters in the larger space
	classLabels = VectorXi(simGraph.numberOfVertices());

	for (int i = 0; i < simGraph.numberOfVertices(); i++) {
		classLabels(i) = labels.at<int>(i,0);
	}
}