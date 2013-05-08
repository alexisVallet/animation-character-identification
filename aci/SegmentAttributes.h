/** @file */
#pragma once

#pragma once

#include <opencv2/opencv.hpp>
#include "WeightedGraph.hpp"
#include "LabeledGraph.hpp"
#include "DisjointSet.hpp"

/**
 * Datatype describing graph labeling functions.
 */
typedef void (*Labeling)(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph);

/**
 * Adds color histogram labels to a segmentation graph.
 *
 * @param image image to compute histograms from.
 * @param mask mask of pixels to take into account.
 * @param segmentation a segmentation of the image
 * @param segmentationGraph segmentation graph to add labels to
 * @param binsPerChannel the number of histogram bins per color channel
 */
void colorHistogramLabels(
 Mat_<Vec<uchar,3> > &image,
 DisjointSetForest &segmentation,
 LabeledGraph<Mat> &segmentationGraph,
 int binsPerChannel);

/**
 * Labels vertices of a segmentation graph by the average color of the segment.
 *
 * @param image image to compute average colors from.
 * @param mask mask of pixels to take into account.
 * @param segmentation a segmentation of the image.
 * @param segmentationGraph graph to add labels to.
 */
void averageColorLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph);

/**
 * Labels vertices of a segmentation graph by the gravity center of the segment.
 *
 * @param image image to compute gravity centers from.
 * @param mask mask of pixels to take into account.
 * @param segmentation a segmentation of the image.
 * @param segmentationGraph graph to add labels to.
 */
void gravityCenterLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph);

/**
 * Labels vertices of a segmentation graph by the covariance matrix of the pixel
 * positions of the segment, flattened in a column vector.
 *
 * @param image image to compute covariance matrices from.
 * @param mask mask of pixels to take into account.
 * @param segmentation a segmentation of the image.
 * @param segmentationGraph graph to add labels to.
 */
void pixelsCovarianceMatrixLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph);

/**
 * Labels vertices of a segmentation graph by the index of the label in the graph.
 *
 * @param image to compute labels from.
 * @param mask mask of pixels to take into account.
 * @param segmentation a segmentation of the image.
 * @param segmentationGraph graph to add labels to.
 */
void segmentIndexLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph);

/**
 * Labels vertices of a segmentation graph by the size of the segment relative to
 * the entire size of the character.
 *
 * @param image to compute relative sizes from.
 * @param mask mask of pixels to take into account.
 * @param segmentation a segmentation of the image.
 * @param segmentationGraph graph to add labels to.
 */
void segmentSizeLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph);

/**
 * Concatenates the results of multiple labelling functions into one. Assumes the labels
 * are column vector.
 *
 * @param labelings labelings to concatenate the results of.
 * @param image image to compute labels from.
 * @param mask mask of pixels to take into account.
 * @param segmentation a segmentation of the image.
 * @param segmentationGraph graph to add labels to.
 */
void concatenateLabelings(const vector<Labeling> &labelings, const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph);