#include "SpectralClustering.h"

static void spectralEmbedding_(const WeightedGraph &simGraph, SparseRepresentation matRep, int k, MatrixXd &embeddings, bool normalize = false, bool symmetric = true) {
	assert(k >= 1);

	// compute the matrix representation of the graph
	SparseMatrix<double> rep = matRep(simGraph, bidirectional);

	// compute its k smallest eigenvectors
	VectorXd eigenvalues;

	symmetricSparseEigenSolver(rep, "SM", k, simGraph.numberOfVertices(), eigenvalues, embeddings);

	// normalize embedding coordinates if necessary
	if (normalize) {
		for (int i = 0; i < simGraph.numberOfVertices(); i++) {
			embeddings.row(i).normalize();
		}
	}
}

void spectralClustering(const MatrixXd &similarity, SimilarityGraphRepresentation &graphRep, SparseRepresentation matRep, int k, VectorXi &classLabels, bool normalize, bool symmetric) {
	WeightedGraph simGraph;

	graphRep(similarity, simGraph);

	MatrixXd eigenvectors;

	spectralEmbedding_(simGraph, matRep, k, eigenvectors, normalize, symmetric);

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

void spectralEmbedding(const MatrixXd &similarity, SimilarityGraphRepresentation &graphRep, SparseRepresentation matRep, int k, MatrixXd &embeddings, bool normalize, bool symmetric) {
	WeightedGraph simGraph;

	graphRep(similarity, simGraph);

	spectralEmbedding_(simGraph, matRep, k, embeddings, normalize, symmetric);
}



NeighborhoodGraph::NeighborhoodGraph(double radius) 
	: radius(radius)
{

}

void NeighborhoodGraph::operator() (const MatrixXd &similarity, WeightedGraph &graph) const {
	graph = WeightedGraph(similarity.rows());

	for (int i = 0; i < similarity.rows(); i++) {
		for (int j = i + 1; j < similarity.rows(); j++) {
			if (similarity(i,j) <= this->radius) {
				graph.addEdge(i,j,1);
			}
		}
	}
}

KNearestGraph::KNearestGraph(int k)
	: k(k)
{

}

class IsMoreSimilar {
private:
	const int reference;
	const MatrixXd *similarity;

public:
	IsMoreSimilar(int reference, const MatrixXd *similarity)
		: reference(reference), similarity(similarity)
	{

	}

	bool operator() (int s1, int s2) {
		return (*similarity)(s1, reference) > (*similarity)(s2, reference);
	}
};

void KNearestGraph::operator() (const MatrixXd &similarity, WeightedGraph &graph) const {
	assert(similarity.rows() > k);
	MatrixXd adjMat = MatrixXd::Zero(similarity.rows(), similarity.rows());
	graph = WeightedGraph(similarity.rows());

	for (int i = 0; i < similarity.rows(); i++) {
		// compute the k nearest neighbor of this sample
		IsMoreSimilar comp(i, &similarity);
		priority_queue<int, vector<int>, IsMoreSimilar> kNearest(comp);

		for (int j = 0; j < similarity.rows(); j++) {
			if (j != i) {
				kNearest.push(j);

				if (kNearest.size() > k) {
					kNearest.pop();
				}
			}
		}

		// add the corresponding edges to the graph
		// if they haven't already been added the other
		// way.
		for (int j = 0; j < k; j++) {
			int neighbor = kNearest.top();
			kNearest.pop();

			if (adjMat(neighbor, i) < 1) {
				adjMat(i, neighbor) = 1;

				graph.addEdge(i, neighbor, similarity(i, neighbor));
			}
		}
	}
}

MutualKNearestGraph::MutualKNearestGraph(int k)
	: k(k)
{

}

void MutualKNearestGraph::operator() (const MatrixXd &similarity, WeightedGraph &graph) const {
	assert(similarity.rows() > k);
	MatrixXd adjMat = MatrixXd::Zero(similarity.rows(), similarity.rows());
	graph = WeightedGraph(similarity.rows());

	for (int i = 0; i < similarity.rows(); i++) {
		// compute the k nearest neighbor of this sample
		IsMoreSimilar comp(i, &similarity);
		priority_queue<int, vector<int>, IsMoreSimilar> kNearest(comp);

		for (int j = 0; j < similarity.rows(); j++) {
			if (j != i) {
				kNearest.push(j);

				if (kNearest.size() > k) {
					kNearest.pop();
				}
			}
		}

		// add the corresponding edges to the graph
		// if we have already added it the other way
		for (int j = 0; j < k; j++) {
			int neighbor = kNearest.top();
			kNearest.pop();

			if (adjMat(neighbor, i) < 1) {
				adjMat(i, neighbor) = 1;
			} else {
				graph.addEdge(i, neighbor, similarity(i, neighbor));
			}
		}
	}
}

CompleteGraph::CompleteGraph() {

}

void CompleteGraph::operator() (const MatrixXd &similarity, WeightedGraph &graph) const {
	graph = WeightedGraph(similarity.rows(), similarity.rows() - 1);

	for (int i = 0; i < similarity.rows(); i++) {
		for (int j = i + 1; j < similarity.rows(); j++) {
			graph.addEdge(i,j,similarity(i, j));
		}
	}
}
