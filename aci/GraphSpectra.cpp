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

void packedStorageNormalizedLaplacian(const WeightedGraph &graph, double *L) {
	cout<<"initializing every coefficient to 0"<<endl;
	// we intialize everything to 0
	int n = graph.numberOfVertices();

	for (int i = 0; i < n * (n-1)/2; i++) {
		L[i] = 0;
	}
	cout<<"initializing diagonal"<<endl;
	// we initialize the diagonal to 1
	for (int i = 0; i < graph.numberOfVertices(); i++) {
		int index = toUpperTriangularPacked(i,i);

		if (index < 0 || index >= n * (n + 1) / 2) {
			cout<<"index out of bounds "<<index<<" at ("<<i<<","<<i<<")"<<endl;
		}

		L[index] = 1;
	}

	cout<<"computing degrees"<<endl;
	// we then compute the degrees
	vector<int> degrees(graph.numberOfVertices(), 0);

	for (int i = 0; i < graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		degrees[edge.source] += edge.weight;
		degrees[edge.destination] += edge.weight;
	}

	cout<<"computing non diagonal elements"<<endl;
	// then we compute the non diagonal elements
	for (int i = 0; i < graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		L[toUpperTriangularPacked(edge.source, edge.destination)] = -edge.weight / sqrt((double)degrees[edge.source] * degrees[edge.destination]);
	}
}
