#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/Dense>
#include <math.h>

#include "WeightedGraph.hpp"
#include "DisjointSet.hpp"
#include "GraphSpectra.h"
#include "GraphPartitions.h"
#include "ImageGraphs.h"

using namespace cv;
using namespace std;
using namespace Eigen;

/**
 * Partitions the graph using the normalized cuts method.
 *
 * @param graph graph to partition.
 * @param stop minimal normalized ratio acting a stopping criteria.
 */
DisjointSetForest normalizedCuts(const WeightedGraph &graph, double stop);

/**
 * Segments an image by the normalized cuts method.
 *
 * @param image image to segment.
 * @param mask mask representing pixels of the image to consider for segmentation.
 * @param stop stopping criteria for segmentation, between 0 and 2. Closer to 0 means
 * more segments.
 */
DisjointSetForest normalizedCutsSegmentation(const Mat_<Vec<uchar,3> > &image, const Mat_<float> &mask, double stop);