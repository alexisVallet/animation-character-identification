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
 * @param image image to segment (as returned by imread for instance)
 * @param mask mask of the image specifying pixels to take into account.
 * @param segmentation segmentation of the image.
 * @param segGraph segmentation graph of the image, where vertices are segment
 * and vertices have an edge between them iff the corresponding segment are adjacent.
 */
void segment(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segGraph, int felzenszwalbScale);
