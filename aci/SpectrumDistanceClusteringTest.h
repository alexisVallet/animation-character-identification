#pragma once

#include <iostream>
#include <fstream>

#include "SpectrumDistanceClustering.h"
#include "Segmentation.h"
#include "Utils.hpp"
#include "PreProcessing.h"

SparseMatrix<double> unnormalized(const WeightedGraph &graph, bool bidirectional);

SparseMatrix<double> normalizedSymmetric(const WeightedGraph &graph, bool bidirectional);

SparseMatrix<double> normalizedRandomWalk(const WeightedGraph &graph, bool bidirectional);

void testSpectrumDistanceClustering(ofstream &out, SparseRepresentation matRep, bool symmetric, const vector<pair<Mat_<Vec3b>, Mat_<float> > > &dataSet, const vector<WeightedGraph> &segGraphs, vector<DisjointSetForest> &segmentations);
