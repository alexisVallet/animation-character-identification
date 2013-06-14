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
 * Type specifying which type of scale measure to use for segment sizes, either
 * cardinality (the number of element for finite graphs) or volume (the sum of
 * degrees) of the vertex set.
 */
enum ScaleType { CARDINALITY, VOLUME };

/**
 * Segments a graph using Felzenszwalb's method. Returns the result as
 * a disjoint set forest data structure. Also goes through a post processing
 * phase to weed out small components.
 *
 * @param k scale parameter.
 * @param graph the graph to segment.
 * @param minCompSize minimum size of components.
 * @param scaleType type of scale measure to use for segment size in the threshold
 * function. This is defined as cardinality in the original paper by Felzenszwalb,
 * but we introduce volume as a way of making the algorithm sensitive to local
 * scale.
 * @return a segmentation of the graph.
 */
DisjointSetForest felzenszwalbSegment(int k, WeightedGraph graph, int minCompSize, Mat_<float> mask, ScaleType scaleType = CARDINALITY);

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
