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

SparseMat_<double> sparseLaplacian(const WeightedGraph &graph, bool bidirectional) {
	int dims[2] = {graph.numberOfVertices(), graph.numberOfVertices()};
	SparseMat_<double> result(2, dims);

	if (!bidirectional) {
		for (int i = 0; i < (int)graph.getEdges().size(); i++) {
			Edge edge = graph.getEdges()[i];
		
			result.ref(edge.source, edge.destination) -= edge.weight;
			result.ref(edge.source, edge.source) += edge.weight;

			if (edge.source != edge.destination) {
				result.ref(edge.destination, edge.source) -= edge.weight;
				result.ref(edge.destination, edge.destination) += edge.weight;
			}
		}
	} else {
		for (int i = 0; i < (int)graph.getEdges().size(); i++) {
			Edge edge = graph.getEdges()[i];

			// if this is not a loop, we set (src,dst) only as (dst,src)
			// will be taken care of by the reverse edge, and we increment
			// the degree of src only so the degree of dst will be incremented
			// by the second edge.
			if (edge.source != edge.destination) {
				result.ref(edge.source, edge.destination) -= edge.weight;
				result.ref(edge.source, edge.source) += edge.weight;
			} 
			// if it is a loop, it has no effect on the laplacian by definition
			// of the degree.
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
