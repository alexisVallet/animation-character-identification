/** @file */
#pragma once

#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "WeightedGraph.hpp"
#include "LabeledGraph.hpp"
#include "DisjointSet.hpp"
#include "GraphPartitions.h"

#define _USE_MATH_DEFINES
#include <math.h>

using namespace Eigen;

/**
 * Datatype for segment labelling function associating vectors
 * to each segment.
 */
typedef vector<VectorXd> (*SegmentLabeling)(DisjointSetForest &segmentation, const Mat_<Vec3b> &image, const Mat_<float> &mask);

/**
 * Labels segments by their average color.
 */
vector<VectorXd> averageColorLabeling(DisjointSetForest &segmentation, const Mat_<Vec3b> &image, const Mat_<float> &mask);

/**
 * Labels segments by their gravity center.
 */
vector<VectorXd> gravityCenterLabeling(DisjointSetForest &segmentation, const Mat_<Vec3b> &image, const Mat_<float> &mask);

/**
 * Labels segments by their area in number of pixels.
 */
vector<VectorXd> segmentAreaLabeling(DisjointSetForest &segmentation, const Mat_<Vec3b> &image, const Mat_<float> &mask);
