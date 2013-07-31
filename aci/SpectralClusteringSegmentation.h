#pragma once

#include <opencv2/opencv.hpp>

#include "SpectralClustering.h"
#include "DisjointSet.hpp"

using namespace std;
using namespace cv;

/**
 * Similarity matrix between pixels of the image, measuring similarity
 * in a feature space including both position and color information.
 */
class PixelSimilarityMatrix : public SimilarityMatrix {
private:
	Mat_<Vec3b> image;
	double colorSigma;
	double positionSigma;

public:
	/**
	 * Defines the similarity matrix on an image, with specific
	 * scale parameters for color and position of pixels.
	 *
	 * @param image image to compute similarities from.
	 * @param colorSigma scale parameter for color gaussian kernel.
	 * @param positionSigma scale parameter for position gaussian kernel.
	 */
	PixelSimilarityMatrix(const Mat_<Vec3b> &image, double colorSigma, double positionSigma);
	double operator() (int i, int j) const;
	int rows() const;
	int cols() const;
};

/**
 * Segments the image into a specific number of segments using spectral
 * clustering.
 */
DisjointSetForest spectralClusteringSegmentation(const Mat_<Vec3b> &image, const Mat_<float> &mask, int nbSegments);
