#pragma once

#include <opencv2/opencv.hpp>

#include "Utils.hpp"
#include "PreProcessing.h"
#include "Segmentation.h"
#include "TWKSpectralClustering.h"

void testTWKSpectralClustering(ofstream &out, int depth, int arity, SpectralClusteringType clusteringType, const vector<pair<Mat_<Vec3b>, Mat_<float> > > &dataSet, const vector<WeightedGraph> &segGraphs, vector<DisjointSetForest> &segmentations);