#include "GraphSpectra.h"

Mat_<double> laplacian(const WeightedGraph &graph) {
	Mat_<double> result = Mat_<double>::zeros(graph.numberOfVertices(), graph.numberOfVertices());

	for (int i = 0; i < (int)graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];
		
		result(edge.source, edge.destination) -= edge.weight;
		result(edge.source, edge.source) += edge.weight;

		if (edge.source != edge.destination) {
			result(edge.destination, edge.source) -= edge.weight;
			result(edge.destination, edge.destination) += edge.weight;
		}
	}

	return result;
}

Mat_<double> normalizedLaplacian(const WeightedGraph &graph) {
	Mat_<double> unnormalized = laplacian(graph);
	Mat_<double> degrees = Mat_<double>::zeros(graph.numberOfVertices(), graph.numberOfVertices());

	for (int i = 0; i < (int)graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		degrees(edge.source, edge.source) += edge.weight;
		degrees(edge.destination, edge.destination) += edge.weight;
	}

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		degrees(i,i) = 1/sqrt(degrees(i,i));
	}

	return degrees * unnormalized * degrees;
}
