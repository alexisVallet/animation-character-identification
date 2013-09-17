#include "SimilarityGraphs.h"

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
	set<pair<int,int> > adjMat;
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

			if (adjMat.find(pair<int,int>(neighbor,i)) == adjMat.end()) {
				adjMat.insert(pair<int,int>(i, neighbor));

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
