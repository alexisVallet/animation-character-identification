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

double khi2Kernel(int binsPerChannel, float lambda, float mu, float gamma, int area1, const Mat &h1, int area2, const Mat &h2) {
	double d2 = 0;

	for (int r = 0; r < binsPerChannel; r++) {
		for (int g = 0; g < binsPerChannel; g++) {
			for (int b = 0; b < binsPerChannel; b++) {
				int pi = h1.at<int>(r,g,b);
				int qi = h2.at<int>(r,g,b);

				if (pi + qi != 0) {
					double toAdd = pow((float)(pi-qi),2) / (float)(pi + qi);
					
					d2 += toAdd;
				}
			}
		}
	}

	double result = lambda * pow(area1, gamma) * pow(area2, gamma) * exp(-mu * d2);

	return result;
}

WeightedGraph weighEdgesByKernel(const MatKernel &kernel, LabeledGraph<Mat> unweightedGraph) {
	WeightedGraph weighted(unweightedGraph.numberOfVertices());

	for (int i = 0; i < unweightedGraph.getEdges().size(); i++) {
		Edge edge = unweightedGraph.getEdges()[i];
		double weight = kernel(unweightedGraph.getLabel(edge.source), unweightedGraph.getLabel(edge.destination));
		weighted.addEdge(edge.source, edge.destination, weight);
	}

	return weighted;
}

CompoundGaussianKernel::CompoundGaussianKernel(Mat_<int> borderLengths)
	: borderLengths(borderLengths)
{

}

double CompoundGaussianKernel::operator() (const Mat &h1, const Mat &h2) const {
	assert(h1.rows == 7 && h2.rows == 7);
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
	Mat E1 = h1.rowRange(6,8);
	Mat E2 = h2.rowRange(6,8);
	
	double cres = exp(-muC * pow(norm(C1, C2), 2) / 3);
	double xres = exp(-muX * pow(norm(X1, X2), 2) / 2);
	double sres = exp(-muS * pow(norm(E1, E2), 2) / 2);

	return pow(this->borderLengths(i,j), gammaB) * (alphaC * cres + alphaX * xres + alphaS * sres);
}