/** @file */
#pragma once

#include <opencv2/opencv.hpp>

#include "WeightedGraph.hpp"
#include "Utils.hpp"
#include "ImageGraphs.h"
#include "SegmentationGraph.hpp"
#include "SegmentAttributes.h"
#include "Felzenszwalb.hpp"
#include "KuwaharaFilter.h"

using namespace std;
using namespace cv;

/**
 * Segmentation method which compute a graph from an animation character image,
 * without background - the background pixels to ignore are specified as 0 in
 * a mask.
 *
 * @param bgrImage BGR image to segment (as returned by imread for instance)
 * @param mask mask of the image specifying pixels to take into account.
 * @return a segmentation graph of the image, where vertices are segments and
 * there is an edge between each adjacent segment in the image. Edges are weighted
 * by similarity between segments.
 */
WeightedGraph computeGraphFrom(Mat_<Vec<uchar,3> > &bgrImage, Mat_<float> &mask);
