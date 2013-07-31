#include "SpectralClustering.h"

#define DEBUG_SEPCTRALCLUSTERING false

DenseSimilarityMatrix::DenseSimilarityMatrix(const MatrixXd *m) 
	: m(m)
{

}

double DenseSimilarityMatrix::operator() (int i, int j) const {
	return (*this->m)(i,j);
}

int DenseSimilarityMatrix::rows() const {
	return this->m->rows();
}

int DenseSimilarityMatrix::cols() const {
	return this->m->cols();
}

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

NeighborhoodGraph::NeighborhoodGraph(double radius) 
	: radius(radius)
{

}

void NeighborhoodGraph::operator() (SimilarityMatrix &similarity, WeightedGraph &graph) const {
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
	const SimilarityMatrix *similarity;

public:
	IsMoreSimilar(int reference, const SimilarityMatrix *similarity)
		: reference(reference), similarity(similarity)
	{

	}

	bool operator() (int s1, int s2) {
		return (*similarity)(s1, reference) > (*similarity)(s2, reference);
	}
};

void KNearestGraph::operator() (SimilarityMatrix &similarity, WeightedGraph &graph) const {
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

void MutualKNearestGraph::operator() (SimilarityMatrix &similarity, WeightedGraph &graph) const {
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

void CompleteGraph::operator() (SimilarityMatrix &similarity, WeightedGraph &graph) const {
	graph = WeightedGraph(similarity.rows(), similarity.rows() - 1);

	for (int i = 0; i < similarity.rows(); i++) {
		for (int j = i + 1; j < similarity.rows(); j++) {
			graph.addEdge(i,j,similarity(i, j));
		}
	}
}

MaskedSimilarityMatrix::MaskedSimilarityMatrix(const SimilarityMatrix const *internalMatrix, const vector<int> const *indexes)
	: internalMatrix(internalMatrix), indexes(indexes)
{

}

double MaskedSimilarityMatrix::operator() (int i, int j) const {
	return (*this->internalMatrix)((*this->indexes)[i], (*this->indexes)[j]);
}

int MaskedSimilarityMatrix::rows() const {
	return this->indexes->size();
}

int MaskedSimilarityMatrix::cols() const {
	return this->indexes->size();
}

MaskedGraph::MaskedGraph(vector<bool> mask, SimilarityGraphRepresentation *internalRep) 
	: mask(mask), internalRep(internalRep)
{

}

void MaskedGraph::operator() (SimilarityMatrix &similarity, WeightedGraph &graph) const {
	assert(this->mask.size() == similarity.rows());
	// compute indexes of points to take into account
	cout<<"computing indexes"<<endl;
	vector<int> indexes;
	indexes.reserve(this->mask.size());
	int k = 0;

	for (int i = 0; i < (int)this->mask.size(); i++) {
		if (this->mask[i]) {
			indexes.push_back(k);
			k++;
		}
	}

	// call the internal representation with the masked matrix
	cout<<"calling internal representation constructor"<<endl;
	MaskedSimilarityMatrix maskedSim(&similarity, &indexes);

	cout<<"calling internal representation"<<endl;
	(*this->internalRep)(maskedSim, graph);
	cout<<"successful"<<endl;
}
