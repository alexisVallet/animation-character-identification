#include "IsoperimetricGraphPartitioningTest.h"

void testRemoveIsolatedVertices(WeightedGraph &testBidirectional) {
	WeightedGraph newGraph = removeIsolatedVertices(testBidirectional);

	for (int i = 0; i < newGraph.numberOfVertices(); i++) {
		assert(!newGraph.getAdjacencyList(i).empty());
	}
}

void testIsoperimetricGraphPartitioning() {
	WeightedGraph testBidirectional(6);
	int edges[7][2] = {{0,1},{0,2},{2,1},{2,3},{3,4},{4,5},{5,3}};

	for (int i = 0; i < 7; i++) {
		testBidirectional.addEdge(edges[i][0], edges[i][1], 1);
		testBidirectional.addEdge(edges[i][1], edges[i][0], 1);
	}

	cout<<"testing function removeIsolatedVertices"<<endl;
	testRemoveIsolatedVertices(testBidirectional);
	cout<<"passed"<<endl;

	vector<int> vertexMap;
	WeightedGraph connected = removeIsolatedVertices(testBidirectional, vertexMap);

	cout<<"testing isoperimetric graph partitioning"<<endl;
	DisjointSetForest partition = isoperimetricGraphPartitioning(connected, 0.5, 4);
	cout<<"passed"<<endl;
	cout<<partition<<endl;
}
