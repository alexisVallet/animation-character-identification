//
//  ContourDetectionforAnime.h
//  animation-charactor-identification for Mac
//
//  Created by Yuki NAKAGAWA on 2013/05/07.
//  Copyright (c) 2013年 Yuki & Alexis. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <stack>

using namespace std;
using namespace cv;

void ContourDetectionforAnime(Mat_<Vec3b> &src, float threshold_a, float threshold_b, int maxContourWidth);
