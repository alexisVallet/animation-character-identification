#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

#include "DisjointSet.hpp"
#include "WeightedGraph.hpp"
#include "Felzenszwalb.hpp"
#include "SegmentationGraph.hpp"
#include "TreeWalkKernel.hpp"
#include "GraphSpectra.h"
#include "SpectrumDistanceClassifier.h"
#include "Kernels.h"
#include "SegmentAttributes.h"
#include "KuwaharaFilter.h"

void testSpectrumDistanceClassifier();