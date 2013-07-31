#include "SpectralClusteringTest.h"

#define SIGMA ((double)1)

static SparseMatrix<double> _sparseLaplacian(const WeightedGraph &graph, bool bidirectional) {
	VectorXd degrees;

	return sparseLaplacian(graph, false, degrees);
}

static double simFunc(const VectorXd &s1, const VectorXd &s2) {
	return exp(-(s1 - s2).squaredNorm() / pow(SIGMA, 2));
}

void testSpectralClustering() {
}
