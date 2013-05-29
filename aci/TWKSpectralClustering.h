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

static SparseMatrix<double> sparseLaplacian_(const WeightedGraph &graph, bool bidirectional) {
	VectorXd degrees;

	return sparseLaplacian(graph, bidirectional, degrees);
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

public:
	/**
	 * Initializes the clustering algorithm with a specific kernel function for
	 * the tree walk kernel computation, and a specific dataset to compute it from.
	 *
	 * @param kernelFunc kernel function (and associated graph labeling) to use
	 * for tree walk kernel computation.
	 */
	TWKSpectralClustering(const vector<pair<Mat_<Vec3b>, Mat_<float> > > &dataSet, const MatKernel<_Tp, m, n> *kernelFunc, int depth, int arrity)
		: dataSet(dataSet), kernelFunc(kernelFunc), depth(depth), arity(arity)
	{

	}

	/**
	 * Computes the embeddings in outDim dimensional space for the images segmentation
	 * graphs, using only the first part of the spectral clustering algorithm
	 * (eg. before the K-means step).
	 *
	 * @param dataSet image and masks to embed in outDim dimensional space.
	 * @param outDim dimension of the output space, ie. the number of columns of
	 * the embeddings matrix.
	 * @param embeddings output matrix of size n by outDim where n is the number of
	 * image in the dataset. Contains embeddings of input images as rows of the
	 * matrix.
	 */
	void embed(vector<DisjointSetForest> &segmentations, const vector<LabeledGraph<Matx<_Tp, m, n> > > &samples, int outDim, MatrixXd &embeddings) {
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
		MatrixXd S(samples.size(), samples.size());

		for (int i = 0; i < samples.size(); i++) {
			for (int j = i; j < samples.size(); j++) {
				S(i,j) = treeWalkKernel<_Tp, m, n>(*this->kernelFunc, this->depth, this->arity, samples[i], centers[i], samples[j], centers[j]);
				S(j,i) = S(i,j);
			}
		}

		cout<<"computing spectral embedding"<<endl;
		spectralEmbedding(S, KNearestGraph(10), sparseLaplacian_, outDim, embeddings, false, true);
	}

	/**
	 * Clusters the images through spectral clustering, with similarity computed
	 * by tree walk kernels of segmentation graphs.
	 *
	 * @param dataSet images and masks to cluster.
	 * @param classLabels output vector of class labels.
	 */
	//void cluster(const vector<LabeledGraph<Matx<_Tp, m, n> > > &samples, VectorXi &classLabels);
};
