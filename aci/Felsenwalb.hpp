/**
 * Implemtnation of Felsenwalb's segmentation method.
 */
#ifndef _FELSENWALB_HPP_
#define _FELSENWALB_HPP_

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
 * Segments a graph using Felsenwalb's method. Returns the result as
 * a disjoint set forest data structure.
 *
 * @param k scale parameter.
 * @param graph the graph to segment.
 * @return a segmentation of the graph.
 */
DisjointSetForest felsenwalbSegment(int k, WeightedGraph graph);

/**
 * Returns the grid graph of a monochrome image. The grid graph is defined
 * as the graph where vertices are pixels of the image and vertices have an
 * edge (undirected) between them iff they are neighbor in an 8-connectivity
 * sense. Does not repeat edges in both directions, every edge only appears
 * on the first pixel's adjacency list in row major order. Every edge is
 * weighted by the absolute difference of pixel intensity.
 */
WeightedGraph gridGraph(Mat_<uchar> &image);

#endif
