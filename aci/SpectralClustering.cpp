#include "SpectralClustering.h"

void spectralClustering(const MatrixXd &samples, SimilarityFunction &simFunc, SimilarityGraphRepresentation &graphRep, SparseRepresentation matRep, int k, VectorXi &classLabels, bool normalize, bool symmetric) {
	WeightedGraph graph;

	graphRep(samples, simFunc, graph);

	spectralClustering(graph, matRep, k, classLabels, normalize, symmetric, false);
}

void spectralClustering(const WeightedGraph &simGraph, SparseRepresentation matRep, int k, VectorXi &classLabels, bool normalize, bool symmetric, bool bidirectional) {
	assert(k >= 1);
	
	// compute the matrix representation of the graph
	SparseMatrix<double> rep = matRep(simGraph, bidirectional);

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

NeighborhoodGraph::NeighborhoodGraph(double radius) 
	: radius(radius)
{

}

void NeighborhoodGraph::operator() (const MatrixXd &samples, SimilarityFunction &simFunc, WeightedGraph &graph) const {
	graph = WeightedGraph(samples.rows());

	for (int i = 0; i < samples.rows(); i++) {
		for (int j = i + 1; j < samples.rows(); j++) {
			if (simFunc(samples.row(i), samples.row(j)) <= this->radius) {
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
	const MatrixXd *samples;
	int reference;
	SimilarityFunction *simFunc;

public:
	IsMoreSimilar(const MatrixXd *samples, int reference, SimilarityFunction &simFunc)
		: samples(samples), reference(reference), simFunc(&simFunc)
	{

	}

	bool operator() (int s1, int s2) {
		return (*simFunc)(samples->row(s1), samples->row(reference)) > (*simFunc)(samples->row(s2), samples->row(reference));
	}
};

void KNearestGraph::operator() (const MatrixXd &samples, SimilarityFunction &simFunc, WeightedGraph &graph) const {
	assert(samples.rows() > k);
	MatrixXd adjMat = MatrixXd::Zero(samples.rows(), samples.rows());
	graph = WeightedGraph(samples.rows());

	for (int i = 0; i < samples.rows(); i++) {
		// compute the k nearest neighbor of this sample
		IsMoreSimilar comp(&samples, i, simFunc);
		priority_queue<int, vector<int>, IsMoreSimilar> kNearest(comp);

		for (int j = 0; j < samples.rows(); j++) {
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

				graph.addEdge(i, neighbor, simFunc(samples.row(i), samples.row(neighbor)));
			}
		}
	}
}

MutualKNearestGraph::MutualKNearestGraph(int k)
	: k(k)
{

}

void MutualKNearestGraph::operator() (const MatrixXd &samples, SimilarityFunction &simFunc, WeightedGraph &graph) const {
	assert(samples.rows() > k);
	MatrixXd adjMat = MatrixXd::Zero(samples.rows(), samples.rows());
	graph = WeightedGraph(samples.rows());

	for (int i = 0; i < samples.rows(); i++) {
		// compute the k nearest neighbor of this sample
		IsMoreSimilar comp(&samples, i, simFunc);
		priority_queue<int, vector<int>, IsMoreSimilar> kNearest(comp);

		for (int j = 0; j < samples.rows(); j++) {
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
				graph.addEdge(i, neighbor, simFunc(samples.row(i), samples.row(neighbor)));
			}
		}
	}
}

CompleteGraph::CompleteGraph() {

}

void CompleteGraph::operator() (const MatrixXd &samples, SimilarityFunction &simFunc, WeightedGraph &graph) const {
	graph = WeightedGraph(samples.rows(), samples.rows() - 1);

	for (int i = 0; i < samples.rows(); i++) {
		for (int j = i + 1; j < samples.rows(); j++) {
			graph.addEdge(i,j,simFunc(samples.row(i), samples.row(j)));
		}
	}
}
