#include "DirichletEigenSubspaceClassifier.h"

DirichletEigenSubspaceClassifier::DirichletEigenSubspaceClassifier(int nbEigenvectors, double minSubgraphSize, SimilarityType simType) 
	: nbEigenvectors(nbEigenvectors), minSubgraphSize(minSubgraphSize), simType(simType)
{
	assert(nbEigenvectors >= 1);
	assert(minSubgraphSize > 0);
	assert(minSubgraphSize < 1);
}

MatrixXd DirichletEigenSubspaceClassifier::bfsNormalizedLaplacian(const WeightedGraph &graph, int faceVertex) {
	vector<int> bfsOrder = breadthFirstSearch(graph, faceVertex);
	WeightedGraph permuted = permuteVertices(graph, bfsOrder);

	return eigNormalizedLaplacian(permuted);
}

void DirichletEigenSubspaceClassifier::train(vector<std::tuple<WeightedGraph, int, int> > trainingSamples) {
	this->trainingBase.clear();
	this->trainingBase.reserve(trainingSamples.size());

	// for each training sample, compute and store the normalized laplacian
	// along with class label. The face vertex is implicit in the Laplacian,
	// as it is always indexed at the first row/column.
	for (int i = 0; i < trainingSamples.size(); i++) {
		
		std::tuple<WeightedGraph,int,int> sample = trainingSamples[i];
		MatrixXd laplacian = 
			this->bfsNormalizedLaplacian(get<0>(sample), get<2>(sample));

		this->trainingBase.push_back(pair<MatrixXd,int>(laplacian,get<1>(sample)));
	}
}

double DirichletEigenSubspaceClassifier::computeSimilarity(const MatrixXd &l1, const MatrixXd &l2) {
	// We assume that l1 is bigger than l2
	if (l1.rows() < l2.rows()) {
		return this->computeSimilarity(l2, l1);
	}

	// The dirichlet eigenvalues/eigenvectors of the subgraphs are
	// none other than the eigenvalues/eigenvectors of submatrices
	// of the host graph. Using our BFS order, we can just take blocks
	// from (0,0) of the right size in constant time.
	MatrixXd evectors1, evectors2;
	int size = (int)floor(l2.rows() * this->minSubgraphSize);

	SelfAdjointEigenSolver<MatrixXd> 
		solver1(l1.block(0,0,size,size)), 
		solver2(l2.block(0,0,size,size));

	evectors1 = solver1.eigenvectors();
	evectors2 = solver2.eigenvectors();

	MatrixXd U, V;
	VectorXd canonicalCos;

	canonicalAngles(
		evectors1.block(0,0,size,this->nbEigenvectors), 
		evectors2.block(0,0,size,this->nbEigenvectors), 
		U, V, canonicalCos);

	switch (this->simType) {
	case AVERAGE:
		return canonicalCos.sum() / (double)canonicalCos.size();
		break;
	case SMALLEST:
		return canonicalCos.maxCoeff();
		break;
	}
}

static bool secondCompare(pair<int,double> p1, pair<int,double> p2) {
	return p1.second < p2.second;
}

int DirichletEigenSubspaceClassifier::predict(const WeightedGraph &graph, int faceVertex) {
	MatrixXd inputLaplacian = this->bfsNormalizedLaplacian(graph, faceVertex);
	vector<pair<int,double> > similarities;
	similarities.reserve(this->trainingBase.size());

	for (int i = 0; i < (int)this->trainingBase.size(); i++) {
		similarities.push_back(
			pair<int,double>(
				i, 
				this->computeSimilarity(inputLaplacian, this->trainingBase[i].first)));
	}

	return max_element(similarities.begin(), similarities.end(), secondCompare)->second;
}

int faceSegment(int width, DisjointSetForest &segmentation, const Vector2d &facePosition) {
	map<int, int> rootIndexes = segmentation.getRootIndexes();

	return rootIndexes[segmentation.find(toRowMajor(width, (int)facePosition(0), (int)facePosition(1)))];
}