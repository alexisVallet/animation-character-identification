#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "SegmentAttributes.h"

using namespace cv;
using namespace Eigen;
using namespace std;

/**
 * Computes a one to one relation between the segments of two images,
 * where related segments are the most similar in the image. Runs in
 * O(|A||B|) time and O(|A| + |B|) space where A and B are the two
 * segmentations.
 *
 * @param lSeg first segmentation.
 * @param lImage image segmented by lSeg.
 * @param lMask mask of image lImage.
 * @param sSeg first segmentation.
 * @param sImage image segmented by sSeg.
 * @param sMask mask of image sImage.
 * @param labelingFunctions segment labeling functions and associated scale
 * parameters to label segment with, and measure similarity between segments
 * using the gaussian kernel.
 * @param ignoreLast true iff the function must ignore the last segment.
 * Useful when the last segment is the background segment, which is of no
 * interest.
 */
vector<std::tuple<int, int, double> > mostSimilarSegments(
	DisjointSetForest &lSeg, const Mat_<Vec3b> &lImage, const Mat_<float> &lMask,
	DisjointSetForest &sSeg, const Mat_<Vec3b> &sImage, const Mat_<float> &sMask,
	vector<pair<SegmentLabeling, double> > labelingFunctions,
	bool ignoreLast = false);
