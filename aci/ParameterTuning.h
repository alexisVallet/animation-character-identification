#pragma once

#include <opencv2/opencv.hpp>

#include "PreProcessing.h"
#include "Segmentation.h"
#include "SpectrumDistanceClassifier.h"
#include "DatasetIO.h"

using namespace std;
using namespace cv;

void parameterTuning(ostream &outStream);