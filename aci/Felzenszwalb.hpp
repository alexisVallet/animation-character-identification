/** @file */
/**
 * Implemtnation of Felzenszwalb's segmentation method.
 */
#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "WeightedGraph.hpp"
#include "DisjointSet.hpp"
#include "Utils.hpp"
#include "ImageGraphs.h"

using namespace std;
using namespace cv;

/**
 * Segments a graph using Felzenszwalb's method. Returns the result as
 * a disjoint set forest data structure. Also goes through a post processing
 * phase to weed out small components.
 *
 * @param k scale parameter.
 * @param graph the graph to segment.
 * @param minCompSize minimum size of components.
 * @return a segmentation of the graph.
 */
DisjointSetForest felzenszwalbSegment(int k, WeightedGraph graph, int minCompSize, Mat_<float> mask);

/**
 * Combines segmentations of the same graph by the following rule:
 * two neighboring vertices in the graph are in the same component iff
 * they are in the same component in all segmentations. Useful for
 * combining segmentations of separate channels of a color image. 
 *
 * Weights are ignored, only the graph structure is kept - it is assumed the
 * segmentations were done on the same unweighted graph.
 *
 * @param graph the graph segmented by the segmentations
 * @param segmentations segmentations of sourceImage to combine.
 */
DisjointSetForest combineSegmentations(const WeightedGraph &imageGraph, vector<DisjointSetForest> &segmentations);
