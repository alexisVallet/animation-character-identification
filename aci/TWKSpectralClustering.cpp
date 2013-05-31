#include "TWKSpectralClustering.h"

TWBasisKernel::TWBasisKernel(float muC, float muA)
	: muC(muC), muA(muA)
{
	assert(muC >= 0);
	assert(muA >= 0);
}

static void twLabeling(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, const WeightedGraph &segGraph, LabeledGraph<Matx<float, 4, 1> > &labeledGraph) {
	concatenateLabelings<float,3,1,1>(averageColorLabels, segmentAreaLabels, image, mask, segmentation, segGraph, labeledGraph);
}

Labeling<float,4,1>::type TWBasisKernel::getLabeling() const {
	return twLabeling;
}

double TWBasisKernel::operator() (const Matx<float,4,1> &l1, const Matx<float,4,1> &l2) const {
	Mat c1 = Mat(l1).rowRange(0,3);
	Mat c2 = Mat(l2).rowRange(0,3);
	float a1 = l1(3,0);
	float a2 = l2(3,0);

	return exp(-this->muC * pow(norm(c1, c2), 2)) * exp(-this->muA * abs(a1 - a2));
}