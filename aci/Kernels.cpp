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

WeightedGraph weighEdgesByKernel(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, const MatKernel &kernel, const LabeledGraph<Mat> &unweightedGraph) {
	LabeledGraph<Mat> labeled = unweightedGraph;
	kernel.getLabeling()(image, mask, segmentation, labeled);
	WeightedGraph weighted(unweightedGraph.numberOfVertices());

	for (int i = 0; i < (int)labeled.getEdges().size(); i++) {
		Edge edge = labeled.getEdges()[i];
		float weight = (float)kernel(labeled.getLabel(edge.source), labeled.getLabel(edge.destination));
		weighted.addEdge(edge.source, edge.destination, weight);
	}

	return weighted;
}

CompoundGaussianKernel::CompoundGaussianKernel(double alphaC, double alphaX, double alphaS)
	: alphaC(alphaC), alphaX(alphaX), alphaS(alphaS)
{
}

static void labeling(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph) {
	vector<Labeling> labelings;

	labelings.push_back(gravityCenterLabels);
	labelings.push_back(averageColorLabels);
	labelings.push_back(pixelsCovarianceMatrixLabels);

	concatenateLabelings(labelings, image, mask, segmentation, segmentationGraph);
}

Labeling CompoundGaussianKernel::getLabeling() const {
	return labeling;
}

double CompoundGaussianKernel::operator() (const Mat &h1, const Mat &h2) const {
	assert(h1.rows == 8 && h2.rows == 8);
	const double muC = 50;
	const double muX = 50;
	const double muS = 50;
	Mat C1 = h1.rowRange(0, 3);
	Mat C2 = h2.rowRange(0, 3);
	Mat X1 = h1.rowRange(3, 5);
	Mat X2 = h2.rowRange(3, 5);
	Mat S1 = h1.rowRange(5, 7);
	Mat S2 = h2.rowRange(5, 7);
	double a1 = (double)h1.at<float>(7,0);
	double a2 = (double)h2.at<float>(7,0);
	
	double cres = exp(-muC * pow(norm(C1, C2), 2) / 3);
	double xres = exp(-muX * pow(norm(X1, X2), 2) / 2);
	double sres = exp(-muS * pow(norm(S1, S2), 2) / 2);
	double ares = exp(-muS * (abs(a1 - a2) / M_PI));

	double result = this->alphaC * cres + this->alphaX * xres + this->alphaS * sres * ares;

	return result;
}
