/** @file */
#pragma once

#include <Eigen/Dense>

#include "SpectralClustering.h"
#include "TreeWalkKernel.hpp"

using namespace std;
using namespace Eigen;
using namespace cv;

/**
 * This class implements clustering of animation images using similarity information
 * from the tree walk kernels between their segmentation graphs, clustered by
 * spectral clustering.
 */
class TWKSpectralClustering {
public:
	TWKSpectralClustering();

	/**
	 * Computes the embeddings in outDim dimensional space for the images in 
	 * the dataset, using only the first part of the spectral clustering algorithm
	 * (eg. before the K-means step).
	 *
	 * @param dataSet data set of images and respective masks to embed in outDim
	 * dimensional space.
	 * @param outDim dimension of the output space, ie. the number of columns of
	 * the embeddings matrix.
	 * @param embeddings output matrix of size n by outDim where n is the number of
	 * image in the dataset. Contains embeddings of input images as rows of the
	 * matrix.
	 */
	void embed(const vector<pair<Mat_<Vec3b>, Mat_<float> > > &dataSet, int outDim, MatrixXd &embeddings);

	/**
	 * Clusters the images through spectral clustering.
	 */
	void cluster(const vector<pair<Mat_<Vec3b>, Mat_<float> > > &dataSet, VectorXi &classLabels);
};
