/** @file */
#pragma once

#include <opencv2/opencv.hpp>
#include <unordered_map>

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
void segment(const Mat_<Vec3f> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, int felzenszwalbScale = 1000);

/**
 * Converts a segmentation image, where each color corresponds to a segment,
 * into a corresponding disjoint set forest data structure representing
 * the segmentation. Ignores segments smaller than a given size, allowing
 * manual "imperfect" segmentations. Also puts the background segment at
 * the last index, yielding a rows * cols + 1 element segmentation.
 */
DisjointSetForest segmentationImageToSegmentation(const Mat_<Vec3b> &segmentationImage, const Mat_<float> &mask);

/**
 * Loads a segmentation from a file. Assumes the file is a color image,
 * where each segment is filled with the exact same color. Greedily fuses
 * segments of size smaller than a threshold.
 *
 * @param mask mask of the image to load a segmentation for.
 * @param segmentationFilename filename of the segmentation file.
 * @return segmentation of the image loaded from the file.
 */
DisjointSetForest loadSegmentation(Mat_<float> &mask, string segmentationFilename);
