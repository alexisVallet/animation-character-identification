#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "SegmentAttributes.h"

using namespace cv;
using namespace Eigen;
using namespace std;

/**
 * Computes a mapping from one image segmentation to another, associating
 * to each segment in the larger segmentation its most similar one according
 * to some segment labeling functions.
 *
 * @param lSeg larger segmentation.
 * @param lImage image corresponding to the larger segmentation.
 * @param lMask mask corresponding to the larger segmentation.
 * @param sSeg smaller segmentation.
 * @param sImage image corresponding to the smaller segmentation.
 * @param sMask mask corresponding to the smaller segmentation.
 * @param labelingFunctions labeling functions to compute similarity from,
 * with associated scale parameters for the gaussian kernel.
 */
vector<pair<int, double> > mostSimilarSegments(
	DisjointSetForest &lSeg, const Mat_<Vec3b> &lImage, const Mat_<float> &lMask,
	DisjointSetForest &sSeg, const Mat_<Vec3b> &sImage, const Mat_<float> &sMask,
	vector<pair<SegmentLabeling, double> > labelingFunctions);
