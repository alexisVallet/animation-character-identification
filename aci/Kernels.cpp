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

static void labeling1(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, const WeightedGraph &segGraph, LabeledGraph<Matx<float, 5, 1> > &labeledGraph) {
	concatenateLabelings<float,2,3,1>(gravityCenterLabels, averageColorLabels, image, mask, segmentation, segGraph, labeledGraph);
}

static void gkLabeling(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, const WeightedGraph &segGraph, LabeledGraph<Matx<float, 8, 1> > &labeledGraph) {	
	concatenateLabelings<float,5,3,1>(labeling1, pixelsCovarianceMatrixLabels, image, mask, segmentation, segGraph, labeledGraph);
}

CompoundGaussianKernel::CompoundGaussianKernel(double alphaC, double alphaX, double alphaS)
	: alphaC(alphaC), alphaX(alphaX), alphaS(alphaS)
{
}

Labeling<float,8,1>::type CompoundGaussianKernel::getLabeling() const {
	return gkLabeling;
}

double CompoundGaussianKernel::operator() (const Matx<float,8,1> &h1, const Matx<float,8,1> &h2) const {
	const double muC = 50;
	const double muX = 50;
	const double muS = 50;
	Mat dh1 = Mat(h1);
	Mat dh2 = Mat(h2);
	Mat C1 = dh1.rowRange(0, 3);
	Mat C2 = dh2.rowRange(0, 3);
	Mat X1 = dh1.rowRange(3, 5);
	Mat X2 = dh2.rowRange(3, 5);
	Mat S1 = dh1.rowRange(5, 7);
	Mat S2 = dh2.rowRange(5, 7);
	double a1 = (double)h1(7,0);
	double a2 = (double)h2(7,0);
	
	double cres = exp(-muC * pow(norm(C1, C2), 2) / 3);
	double xres = exp(-muX * pow(norm(X1, X2), 2) / 2);
	double sres = exp(-muS * pow(norm(S1, S2), 2) / 2);
	double ares = exp(-muS * (abs(a1 - a2) / M_PI));

	double result = this->alphaC * cres + this->alphaX * xres + this->alphaS * sres * ares;

	return result;
}
