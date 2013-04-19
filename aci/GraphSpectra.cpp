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

Eigen::SparseMatrix<double> sparseLaplacian(const WeightedGraph &graph, bool bidirectional, Eigen::VectorXd &degrees) {
	// we construct the triplet list without the diagonal first, computing the degrees
	// as we do so, then add the degree triplets.
	degrees = Eigen::VectorXd::Zero(graph.numberOfVertices());
	typedef Eigen::Triplet<double> T;
	vector<T> tripletList;

	tripletList.reserve(graph.getEdges().size());

	if (!bidirectional) {
		for (int i = 0; i < (int)graph.getEdges().size(); i++) {
			Edge edge = graph.getEdges()[i];
			
			if (edge.source != edge.destination) {
				tripletList.push_back(T(edge.source, edge.destination, -edge.weight));
				tripletList.push_back(T(edge.destination, edge.source, -edge.weight));
				degrees(edge.source) += edge.weight;
				degrees(edge.destination) += edge.weight;
			} else {
				degrees(edge.source) -= edge.weight;
			}
		}
	} else {
		for (int i = 0; i < (int)graph.getEdges().size(); i++) {
			Edge edge = graph.getEdges()[i];

			if (edge.source != edge.destination) {
				tripletList.push_back(T(edge.source, edge.destination, -edge.weight));
				degrees(edge.source) += edge.weight;
			} else {
				degrees(edge.source) -= edge.weight / 2;
			}
		}
	}

	// add the diagonal degree elements to the triplet list
	for (int i = 0; i < graph.numberOfVertices(); i++) {
		tripletList.push_back(T(i, i, degrees(i)));
	}

	Eigen::SparseMatrix<double> result(graph.numberOfVertices(), graph.numberOfVertices());

	result.setFromTriplets(tripletList.begin(), tripletList.end());

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

Eigen::SparseMatrix<double> normalizedSparseLaplacian(const WeightedGraph &graph) {
	typedef Eigen::Triplet<double> T;
	vector<T> tripletList;

	tripletList.reserve(graph.numberOfVertices() + graph.getEdges().size()/2);

	// first we compute the degrees while initializing the diagonal
	Eigen::VectorXd degrees(graph.numberOfVertices());

	for (int i = 0; i < graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		degrees(edge.source) += edge.weight;
		tripletList.push_back(T(i,i,1));
	}

	// then we compute the rest of the matrix
	for (int i = 0; i < graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		tripletList.push_back(T(edge.source, edge.destination, -edge.weight/sqrt(degrees(edge.source) * degrees(edge.destination))));
	}

	Eigen::SparseMatrix<double> normalized(graph.numberOfVertices(), graph.numberOfVertices());

	normalized.setFromTriplets(tripletList.begin(), tripletList.end());

	return normalized;
}
