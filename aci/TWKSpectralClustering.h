/** @file */
#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "GraphPartitions.h"
#include "SpectralClustering.h"
#include "TreeWalkKernel.hpp"
#include "Felzenszwalb.hpp"
#include "Segmentation.h"

using namespace std;
using namespace Eigen;
using namespace cv;

/**
 * Basis kernel for tree walk kernels applied to animation character images,
 * as defined by (Nakagawa, 2013). Computes similarity using area and average
 * L*a*b* color of segments.
 */
class TWBasisKernel : public LabeledMatKernel<float, 4, 1> {
private:
	const float muC;
	const float muA;

public:
	/**
	 * Initialized the kernel function using parameters for relative importance
	 * of color relative to area. See documentation for operator() for the exact
	 * formula.
	 *
	 * @param muC relative importance of average color.
	 * @param muA relative importance of area.
	 */
	TWBasisKernel(float muC, float muA);
	/**
	 * Labeling function associating average L*a*b* color and segment area
	 * to each vertex.
	 */
	Labeling<float,4,1>::type getLabeling() const;
	/**
	 * Actual kernel function:
	 * k(r,s) = exp(-muC * d(r,s)^2) * exp(-muA * |s(r) - s(s)|)
	 * Where d(r,s) is the euclidean distance between average color or segments
	 * r, s and s(x) is the area of segment x.
	 *
	 * @param l1 first label as computed by getLabeling.
	 * @param l2 second label as computed by getLabeling.
	 * @return some similarity value in the [0;1] range.
	 */
	double operator() (const Matx<float,4,1> &l1, const Matx<float,4,1> &l2) const;
};

static SparseMatrix<double> sparseLaplacian_(const WeightedGraph &graph, bool bidirectional) {
	VectorXd degrees;

	return randomWalkSparseLaplacian(graph, bidirectional, degrees);
}

/**
 * This class implements clustering of animation images using similarity information
 * from the tree walk kernels between their segmentation graphs, clustered by
 * spectral clustering.
 */
template < typename _Tp, int m, int n >
class TWKSpectralClustering {
private:
	const vector<pair<Mat_<Vec3b>, Mat_<float> > > dataSet;
	const int depth;
	const int arity;
	const MatKernel<_Tp, m, n> *kernelFunc;

	void similarityMatrix(vector<DisjointSetForest> &segmentations, const vector<LabeledGraph<Matx<_Tp, m, n> > > &samples, MatrixXd &S) {
		assert(samples.size() == this->dataSet.size());
		assert(segmentations.size() == samples.size());
		cout<<"computing gravity centers"<<endl;
		// compute gravity centers
		vector<vector<Vec2f> > centers(samples.size());

		for (int i = 0; i < samples.size(); i++) {
			gravityCenters(this->dataSet[i].first, this->dataSet[i].second, segmentations[i], centers[i]);
		}

		cout<<"computing similarity matrix"<<endl;
		// compute similarity matrix from all pairs tree walk kernels. This is a 
		// symmetric matrix because the tree walk kernel is a kernel function,
		// assuming the inner kernel is (which we do assume).
		S = MatrixXd(samples.size(), samples.size());

		for (int i = 0; i < samples.size(); i++) {
			for (int j = i; j < samples.size(); j++) {
				S(i,j) = treeWalkKernel<_Tp, m, n>(*this->kernelFunc, this->depth, this->arity, samples[i], centers[i], samples[j], centers[j]);
				S(j,i) = S(i,j);
			}
		}
	}

public:
	/**
	 * Initializes the clustering algorithm with a specific kernel function for
	 * the tree walk kernel computation, and a specific dataset to compute it from.
	 *
	 * @param kernelFunc kernel function (and associated graph labeling) to use
	 * for tree walk kernel computation.
	 */
	TWKSpectralClustering(const vector<pair<Mat_<Vec3b>, Mat_<float> > > &dataSet, const MatKernel<_Tp, m, n> *kernelFunc, int depth, int arity)
		: dataSet(dataSet), kernelFunc(kernelFunc), depth(depth), arity(arity)
	{

	}

	/**
	 * Computes the embeddings in outDim dimensional space for the images segmentation
	 * graphs, using only the first part of the spectral clustering algorithm
	 * (eg. before the K-means step).
	 *
	 * @param segmentations segmentations corresponding to the graphs.
	 * @param samples segmentation graphs to embed.
	 * @param outDim dimension of the output space, ie. the number of columns of
	 * the embeddings matrix.
	 * @param embeddings output matrix of size n by outDim where n is the number of
	 * image in the dataset. Contains embeddings of image segmentations.
	 */
	void embed(vector<DisjointSetForest> &segmentations, const vector<LabeledGraph<Matx<_Tp, m, n> > > &samples, int outDim, MatrixXd &embeddings) {
		MatrixXd S;

		similarityMatrix(segmentations, samples, S);
		spectralEmbedding(S, KNearestGraph(10), sparseLaplacian_, outDim, embeddings, false, false);
	}

	/**
	 * Clusters the images through spectral clustering, with similarity computed
	 * by tree walk kernels of segmentation graphs.
	 *
	 * @param segmentations segmentations corresponding to the graphs.
	 * @param samples segmentation graphs to cluster.
	 * @param nbClasses number of classes to cluster the graphs into.
	 * @param classLabels class label for each graph, in the {0, ..., nbClasses - 1} set.
	 */
	void cluster(vector<DisjointSetForest> &segmentations, const vector<LabeledGraph<Matx<_Tp, m, n> > > &samples, int nbClasses, VectorXi &classLabels) {
		 MatrixXd S;

		similarityMatrix(segmentations, samples, S);
		spectralClustering(S, KNearestGraph(10), sparseLaplacian_, nbClasses, classLabels, false, false);
	}
};
