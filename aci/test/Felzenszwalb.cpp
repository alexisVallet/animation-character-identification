#include <iostream>
#include <opencv2/opencv.hpp>

#include "Felzenszwalb.hpp"
#include "../ACIConfig.h"

using namespace std;
using namespace cv;

int main() {
  Mat raw = imread(ACI_SOURCE_DIR "/aci/test/mickey.png");
  Mat rawLab;
  cvtColor(raw, rawLab, CV_RGB2Lab);
  Mat_<uchar> rawChannels[3];
  split(rawLab, rawChannels);
  Mat_<uchar> smoothChannels[3];
  for (int i = 0; i < 3; i++) {
    GaussianBlur(rawChannels[i], smoothChannels[i], Size(0,0), 0.8);
  }
  Mat_<Vec<uchar,3> > image;
  merge(smoothChannels, 3, image);
  WeightedGraph graph = gridGraph(image, CONNECTIVITY_4);
  DisjointSetForest segmentation = felzenszwalbSegment(150, graph, 50);
  Mat_<Vec<uchar,3> > segImage = segmentation.toRegionImage(image);

  namedWindow("orignal");
  imshow("original", rawLab);
  namedWindow("filtered");
  imshow("filtered", image);
  namedWindow("segmented");
  imshow("segmented", segImage);
  waitKey(0);

  return 0;
}
