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

#define DEBUG_SEGMENTATION false
#define CONNECTIVITY CONNECTIVITY_4
#define MAX_SEGMENTS 250

static bool compareGraphSize(const WeightedGraph& g1, const WeightedGraph& g2) {
	return g1.numberOfVertices() < g2.numberOfVertices();
}

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
template < typename _Tp, int m, int n >
void segment(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Matx<_Tp, m, n> > &segGraph, int felzenszwalbScale) {
	assert(felzenszwalbScale >= 0);
	WeightedGraph grid = gridGraph(image, CONNECTIVITY, mask, euclidDistance, false);
	int minCompSize = countNonZero(mask) / MAX_SEGMENTS;
	segmentation = felzenszwalbSegment(felzenszwalbScale, grid, minCompSize, mask);
	segGraph = segmentationGraph<Matx<_Tp, m, n> >(image, segmentation, grid);

	if (DEBUG_SEGMENTATION) {
		imshow("image", image);
		waitKey(0);
		Mat regionImage = segmentation.toRegionImage(image);
		segGraph.drawGraph(segmentCenters(image, segmentation), regionImage);
		imshow("segmentation graph", regionImage);
		cout<<"number of components: "<<segmentation.getNumberOfComponents()<<endl;
		waitKey(0);
	}

	// keep only the largest connected component
	vector<int> inConnectedComponent;
	int nbCC;

	connectedComponents(segGraph, inConnectedComponent, &nbCC);

	vector<int> vertexIdx;
	vector<WeightedGraph> components;

	inducedSubgraphs(segGraph, inConnectedComponent, nbCC, vertexIdx, components);

	WeightedGraph largest = *max_element(components.begin(), components.end(), compareGraphSize);

	segGraph = LabeledGraph<Matx<_Tp, m, n> >(largest.numberOfVertices());

	segGraph.copyEdges(largest);
}
