#include "SpectralClusteringTest.h"

static SparseMatrix<double> _sparseLaplacian(const WeightedGraph &graph) {
	VectorXd degrees;

	return randomWalkSparseLaplacian(graph, false, degrees);
}

void testSpectralClustering() {
	WeightedGraph testGraph(6, 3);

	testGraph.addEdge(0,2,10);
	testGraph.addEdge(2,4,10);
	testGraph.addEdge(4,0,10);
	testGraph.addEdge(2,1,1);
	testGraph.addEdge(1,3,10);
	testGraph.addEdge(3,5,10);
	testGraph.addEdge(5,1,10);

	VectorXi classLabels;

	spectralClustering(testGraph, _sparseLaplacian, 2, classLabels, false, false);

	cout<<classLabels<<endl;
}