/** @file */
#pragma once

#pragma once

#include <opencv2/opencv.hpp>
#include "WeightedGraph.hpp"
#include "LabeledGraph.hpp"
#include "DisjointSet.hpp"
#include "GraphPartitions.h"

#define _USE_MATH_DEFINES
#include <math.h>

/**
 * Labels vertices of a segmentation graph by the area of the segment.
 *
 * @param image image to compute average colors from.
 * @param mask mask of pixels to take into account.
 * @param segmentation a segmentation of the image.
 * @param segGraph graph to add labels to.
 * @param labeledGraph output graph with labels added.
 */
void segmentAreaLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, const WeightedGraph &segGraph, LabeledGraph<Matx<float,1,1> > &labeledGraph);

/**
 * Labels vertices of a segmentation graph by the average color of the segment.
 *
 * @param image image to compute average colors from.
 * @param mask mask of pixels to take into account.
 * @param segmentation a segmentation of the image.
 * @param segGraph graph to add labels to.
 * @param labeledGraph output graph with labels added.
 */
void averageColorLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, const WeightedGraph &segGraph, LabeledGraph<Matx<float,3,1> > &labeledGraph);

/**
 * Labels vertices of a segmentation graph by the gravity center of the segment.
 *
 * @param image image to compute gravity centers from.
 * @param mask mask of pixels to take into account.
 * @param segmentation a segmentation of the image.
 * @param segGraph graph to add labels to.
 * @param labeledGraph output graph with labels added.
 */
void gravityCenterLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, const WeightedGraph &segGraph, LabeledGraph<Matx<float, 2, 1> > &labeledGraph);

/**
 * Labels vertices of a segmentation graph by the covariance matrix of the pixel
 * positions of the segment, flattened in a column vector.
 *
 * @param image image to compute covariance matrices from.
 * @param mask mask of pixels to take into account.
 * @param segmentation a segmentation of the image.
 * @param segGraph graph to add labels to.
 * @param labeledGraph output graph with labels added.
 */
void pixelsCovarianceMatrixLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, const WeightedGraph &segGraph, LabeledGraph<Matx<float, 3, 1> > &labeledGrpah);

/**
 * Concatenates the results of multiple labelling functions into one. Assumes the labels
 * are column vector.
 *
 * @param l1 labeling to concatenate with l2
 * @param l2 labeling to concatenate with l1
 * @param image image to compute labels from.
 * @param mask mask of pixels to take into account.
 * @param segmentation a segmentation of the image.
 * @param segGraph graph to add labels to.
 * @param labeledGraph output graph with labels added.
 */
template < typename _Tp, int m1, int m2, int n >
void concatenateLabelings(const typename Labeling<_Tp, m1, n>::type l1, const typename Labeling<_Tp, m2, n>::type l2, const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, const WeightedGraph &segGraph, LabeledGraph<Matx<_Tp, m1 + m2, n> > &labeledGraph) {
	// computing first labeling
	LabeledGraph<Matx<_Tp, m1, n> > g1;

	l1(image, mask, segmentation, segGraph, g1);

	// computing second labeling
	LabeledGraph<Matx<_Tp, m2, n> > g2;

	l2(image, mask, segmentation, segGraph, g2);

	labeledGraph = LabeledGraph<Matx<_Tp, m1 + m2, n> >(segGraph.numberOfVertices());
		
	labeledGraph.copyEdges(segGraph);

	for (int i = 0; i < segGraph.numberOfVertices(); i++) {
		Matx<_Tp, m1 + m2, n> newLabel;

		vconcatX<_Tp, m1, m2, n>(g1.getLabel(i), g2.getLabel(i), newLabel);

		labeledGraph.addLabel(i, newLabel);
	}
}

/**
 * Adds a vertex to a labelled graph connected to all other vertices in the original
 * graph, labeled with the 0 matrix with the same size as the other labels.
 *
 * @param graph graph to add a vertex to.
 * @return the same graph as the input with an additional vertex labeled with the 0
 * matrix and connected to all other vertices.
 */
template < typename _Tp, int m, int n >
LabeledGraph<Matx<_Tp, m, n> > addGroundVertex(const LabeledGraph<Matx<_Tp, m, n> > &graph) {
	LabeledGraph<Matx<_Tp, m, n> > newGraph(graph.numberOfVertices() + 1);

	newGraph.copyEdges(graph);

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		newGraph.addLabel(i, graph.getLabel(i));
		newGraph.addEdge(graph.numberOfVertices(), i, 1);
	}

	newGraph.addLabel(graph.numberOfVertices(), Matx<_Tp, m, n>::zeros());

	return newGraph;
}
