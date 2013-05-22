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
	MatrixXd samples(6,2);

	samples.row(0) = Vector2d(0,0);
	samples.row(2) = Vector2d(1,1);
	samples.row(4) = Vector2d(0,2);
	samples.row(1) = Vector2d(5,1);
	samples.row(3) = Vector2d(6,2);
	samples.row(5) = Vector2d(6,0);

	VectorXi classLabels;
	KNearestGraph graphRep(2);

	spectralClustering(samples, simFunc, graphRep, _sparseLaplacian, 2, classLabels, false, true);

	cout<<classLabels<<endl;
}
