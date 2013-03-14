/**
 * Implemtnation of Felzenszwalb's segmentation method.
 */
#ifndef _FELZENSZWALB_HPP_
#define _FELZENSZWALB_HPP_

#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "WeightedGraph.hpp"
#include "DisjointSet.hpp"
#include "Utils.hpp"

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
DisjointSetForest felzenszwalbSegment(int k, WeightedGraph graph, int minCompSize);

/**
 * Returns the grid graph of a color image. The grid graph is defined
 * as the graph where vertices are pixels of the image and vertices have an
 * edge (undirected) between them iff they are neighbor in an N-connectivity
 * sense. Does not repeat edges in both directions, every edge only appears
 * on the first pixel's adjacency list in row major order. Every edge is
 * weighted by euclidean distance between pixel values.
 *
 * @param image image to compute the grid graph from.
 */
WeightedGraph gridGraph(Mat_<Vec<uchar,3> > &image, ConnectivityType connectivity);

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


#endif
