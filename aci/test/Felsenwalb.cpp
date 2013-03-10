#include <iostream>
#include <opencv2/opencv.hpp>

#include "Felsenwalb.hpp"
#include "../ACIConfig.h"

using namespace std;
using namespace cv;

int main() {
  Mat rawMickey = imread(ACI_SOURCE_DIR "/aci/test/mickey.jpg");
  Mat mickey;
  GaussianBlur(rawMickey, mickey, Size(0,0), 0.8);
  Mat_<uchar> mickeyChannels[3];
  split(mickey, mickeyChannels);

  for (int i = 0; i < 3; i++) {
    WeightedGraph graph = gridGraph(mickeyChannels[i]);
    DisjointSetForest forest = felsenwalbSegment(300, graph);
    Mat_<Vec<uchar,3> > regionImage =
      forest.toRegionImage((Mat_<Vec<uchar,3> >)mickey);

    namedWindow("mickey");
    imshow("mickey", mickeyChannels[i]);
    namedWindow("segmentation");
    imshow("segmentation", regionImage);
    waitKey(0);
  }

  return 0;
}
