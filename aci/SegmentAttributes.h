/** @file */
#pragma once

#pragma once

#include <opencv2/opencv.hpp>
#include "WeightedGraph.hpp"
#include "LabeledGraph.hpp"
#include "DisjointSet.hpp"

typedef void (*Labeling)(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph);

void colorHistogramLabels(
 Mat_<Vec<uchar,3> > &image,
 DisjointSetForest &segmentation,
 LabeledGraph<Mat> &segmentationGraph,
 int binsPerChannel);

void averageColorLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph);

void gravityCenterLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph);

void concatenateLabelings(const vector<Labeling> &labelings, const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph);