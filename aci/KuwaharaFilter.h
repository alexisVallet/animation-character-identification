#pragma once

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat_<Vec<uchar,3>> KuwaharaFilter(Mat_<Vec<uchar,3>> &image, uchar size);
