/** @file */
#pragma once

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void KuwaharaFilter(Mat_<Vec<uchar,3>> &src, Mat_<Vec<uchar,3>> &dest, uchar filterSize);
