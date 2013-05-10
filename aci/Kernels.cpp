#include "Kernels.h"

double euclidDistance(const Mat &h1, const Mat &h2) {
	return norm(h1, h2);
}

double dotProductKernel(const Mat &h1, const Mat &h2) {
	return h1.dot(h2);
}

double gaussianKernel(float mu, float sigma, const Mat &h1, const Mat &h2) {
	return exp(- mu * pow(norm(h1, h2), 2) / ( 2 * pow(sigma, 2)));
}

WeightedGraph weighEdgesByKernel(const MatKernel &kernel, LabeledGraph<Mat> unweightedGraph) {
	WeightedGraph weighted(unweightedGraph.numberOfVertices());

	for (int i = 0; i < (int)unweightedGraph.getEdges().size(); i++) {
		Edge edge = unweightedGraph.getEdges()[i];
		float weight = (float)kernel(unweightedGraph.getLabel(edge.source), unweightedGraph.getLabel(edge.destination));
		weighted.addEdge(edge.source, edge.destination, weight);
	}

	return weighted;
}

CompoundGaussianKernel::CompoundGaussianKernel(Mat_<int> borderLengths)
	: borderLengths(borderLengths)
{

}

double CompoundGaussianKernel::operator() (const Mat &h1, const Mat &h2) const {
	assert(h1.rows == 10 && h2.rows == 10);
	const double alphaC = 5;
	const double muC = 50;
	const double alphaX = 5;
	const double muX = 50;
	const double alphaS = 5;
	const double muS = 20;
	const double gammaB = 1;
	Mat C1 = h1.rowRange(0, 3);
	Mat C2 = h2.rowRange(0, 3);
	Mat X1 = h1.rowRange(3, 5);
	Mat X2 = h2.rowRange(3, 5);
	int i = (int)h1.at<float>(5,0);
	int j = (int)h2.at<float>(5,0);
	Mat E1 = h1.rowRange(6,10);
	Mat E2 = h2.rowRange(6,10);
	
	double cres = exp(-muC * pow(norm(C1, C2), 2) / 3);
	double xres = exp(-muX * pow(norm(X1, X2), 2) / 2);
	double sres = exp(-muS * pow(norm(E1, E2), 2) / 2);

	return pow(this->borderLengths(i,j), gammaB) * (alphaC * cres + alphaX * xres + alphaS * sres);
}