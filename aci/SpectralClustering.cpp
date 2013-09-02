#include "SpectralClustering.h"

#define DEBUG_SEPCTRALCLUSTERING false

static void spectralEmbedding_(const WeightedGraph &simGraph, SparseRepresentation matRep, int k, MatrixXd &embeddings, bool normalize = false, bool symmetric = true) {
	assert(k >= 1);

	// compute the matrix representation of the graph
	SparseMatrix<double> rep = matRep(simGraph, false);

	// compute its k smallest eigenvectors
	VectorXd eigenvalues;

	if (symmetric) {
		symmetricSparseEigenSolver(rep, "SM", k + 1, simGraph.numberOfVertices(), eigenvalues, embeddings);
	} else {
		nonSymmetricSparseEigenSolver(rep, "SM", k + 1, simGraph.numberOfVertices(), eigenvalues, embeddings);
	}

	embeddings = embeddings.block(0, 1, simGraph.numberOfVertices(), k);
	
	// normalize embedding coordinates if necessary
	if (normalize) {
		for (int i = 0; i < simGraph.numberOfVertices(); i++) {
			embeddings.row(i).normalize();
		}
	}

	if (DEBUG_SEPCTRALCLUSTERING) {
		cout<<"laplacian: "<<endl<<rep<<endl;
		MatrixXd dense(rep);
		SelfAdjointEigenSolver<MatrixXd> solver(dense);
		cout<<"expected eigenvalues: "<<solver.eigenvalues()<<endl;
		cout<<"expected eigenvectors:"<<endl<<solver.eigenvectors().block(0,1,simGraph.numberOfVertices(),k)<<endl;
		cout<<"actual eigenvalues: "<<eigenvalues<<endl;
		cout<<"actual eigenvectors:"<<endl<<embeddings<<endl<<endl;
	}
}

void spectralClustering(SimilarityMatrix &similarity, SimilarityGraphRepresentation &graphRep, SparseRepresentation matRep, int k, VectorXi &classLabels, bool normalize, bool symmetric) {
	WeightedGraph simGraph;

	cout<<"computing graph representation"<<endl;
	graphRep(similarity, simGraph);

	MatrixXd eigenvectors;

	cout<<"computing spectral embedding"<<endl;
	spectralEmbedding_(simGraph, matRep, k, eigenvectors, normalize, symmetric);

	// cluster the embedding using K-means
	cout<<"clustering using kmeans"<<endl;
	MatrixXf fEigenvectors = eigenvectors.cast<float>();
	Mat_<float> embeddings(simGraph.numberOfVertices(), k);

	for (int i = 0; i < simGraph.numberOfVertices(); i++) {
		for (int j = 0; j < k; j++) {
			embeddings(i,j) = fEigenvectors(i,j);
		}
	}

	Mat labels;

	cout<<embeddings.colRange(0,5)<<endl;

	kmeans(embeddings, k, labels, TermCriteria(), 1, KMEANS_RANDOM_CENTERS);

	// compute the corresponding clusters in the larger space
	classLabels = VectorXi(simGraph.numberOfVertices());

	for (int i = 0; i < simGraph.numberOfVertices(); i++) {
		classLabels(i) = labels.at<int>(i,0);
	}
}

class SelfTuningKernelMatrix : public SimilarityMatrix {
private:
	Mat_<double> samples;
	const int k;
	flann::Index *knnIndex;

public:
	SelfTuningKernelMatrix(const MatrixXd &samples, int k) 
		: k(k)
	{
		eigenToCv(samples, this->samples);
	}

	// TODO
	double operator() (int i, int j)  const {
		assert(false);

		return 0;
	}

	int rows() const {
		return samples.rows;
	}

	int cols() const {
		return samples.rows;
	}
};


void spectralEmbedding(SimilarityMatrix &similarity, SimilarityGraphRepresentation &graphRep, SparseRepresentation matRep, int k, MatrixXd &embeddings, bool normalize, bool symmetric) {
	WeightedGraph simGraph;

	graphRep(similarity, simGraph);

	spectralEmbedding_(simGraph, matRep, k, embeddings, normalize, symmetric);
}
