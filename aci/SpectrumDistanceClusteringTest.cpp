#include "SpectrumDistanceClusteringTest.h"

#define SPECTRUMDIST_NBCLASSES 12



void testSpectrumDistanceClustering(ofstream &out, SparseRepresentation matRep, bool symmetric, const vector<pair<Mat_<Vec3b>, Mat_<float> > > &dataSet, const vector<WeightedGraph> &segGraphs, vector<DisjointSetForest> &segmentations) {
	cout<<"weighing"<<endl;
	CompoundGaussianKernel kernel(5, 5, 5);
	vector<WeightedGraph> weighted;
	weighted.reserve(dataSet.size());

	for (int i = 0; i < (int)dataSet.size(); i++) {
		weighted.push_back(weighEdgesByKernel(dataSet[i].first, dataSet[i].second, segmentations[i], kernel, segGraphs[i]));
	}

	// computing the average number of vertices
	double avgNbVertices = 0;

	for (int i = 0; i < (int)dataSet.size(); i++) {
		avgNbVertices += (double)weighted[i].numberOfVertices() / (double)dataSet.size();
	}

	cout<<"embedding"<<endl;
	SpectrumDistanceClustering clustering(matRep, symmetric, 3);
	MatrixXd embeddings;

	clustering.embed(weighted, embeddings);

	eigenMatToCsv(embeddings, out);
}

SparseMatrix<double> unnormalized(const WeightedGraph &graph, bool bidirectional) {
	VectorXd degrees;

	return sparseLaplacian(graph, bidirectional, degrees);
}

SparseMatrix<double> normalizedSymmetric(const WeightedGraph &graph, bool bidirectional) {
	VectorXd degrees;

	return normalizedSparseLaplacian(graph, bidirectional, degrees);
}

SparseMatrix<double> normalizedRandomWalk(const WeightedGraph &graph, bool bidirectional) {
	VectorXd degrees;

	return randomWalkSparseLaplacian(graph, bidirectional, degrees);
}