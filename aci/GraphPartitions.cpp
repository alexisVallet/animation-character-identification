#include "GraphPartitions.h"

void fusePartitions(const WeightedGraph &graph, vector<int> &inSubgraph, vector<int> &vertexIdx, vector<DisjointSetForest> &partitions, DisjointSetForest &partition) {
	for (int i = 0; i < (int)graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		if (inSubgraph[edge.source] == inSubgraph[edge.destination]) {
			int subIndex = inSubgraph[edge.source];

			if (partitions[subIndex].find(vertexIdx[edge.source]) == partitions[subIndex].find(vertexIdx[edge.destination])) {
				partition.setUnion(edge.source, edge.destination);
			}
		}
	}
}

WeightedGraph removeIsolatedVertices(WeightedGraph &graph, vector<int> &vertexMap) {
	// first we count the number of non-isolated vertices in the graph, filling
	// vertexMap appropriately.
	vertexMap = vector<int>(graph.numberOfVertices());
	int nonIsolated = 0;

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		if (!graph.getAdjacencyList(i).empty()) {
			vertexMap[i] = nonIsolated;
			nonIsolated++;
		} else {
			vertexMap[i] = -1;
		}
	}

	WeightedGraph connected(nonIsolated);

	for (int i = 0; i < (int)graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];
		connected.addEdge(vertexMap[edge.source], vertexMap[edge.destination], edge.weight);
	}

	return connected;
}

DisjointSetForest addIsolatedVertices(WeightedGraph &graph, DisjointSetForest &segmentation, vector<int> &vertexMap) {
	assert(graph.numberOfVertices() == vertexMap.size());
	DisjointSetForest result(graph.numberOfVertices());

	// we first reproduce the segmentation in the larger forest, ignoring isolated vertices
	for (int i = 0; i < (int)graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		if (segmentation.find(vertexMap[edge.source]) == segmentation.find(vertexMap[edge.destination])) {
			result.setUnion(edge.source, edge.destination);
		}
	}

	int firstIsolated = -1;

	// We then fuse the isolated vertices in their own component
	for (int i = 0; i < (int)vertexMap.size(); i++) {
		if (vertexMap[i] < 0) {
			if (firstIsolated < 0) {
				firstIsolated = i;
			} else {
				result.setUnion(firstIsolated, i);
			}
		}
	}

	return result;
}