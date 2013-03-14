#include <opencv2/opencv.hpp>
#include <iostream>

#include "TreeWalkKernel.hpp"
#include "Felzenszwalb.hpp"
#include "SegmentationGraph.hpp"
#include "../ACIConfig.h"

#define BINS_PER_CHANNEL 10

float basisKernel(const Mat &h1, const Mat &h2) {
  return khi2Kernel(BINS_PER_CHANNEL, 0.75, 1, h1, h2);
}

int main() {
  Mat_<Vec<uchar,3> > image = imread(ACI_SOURCE_DIR "/aci/test/mickey.png");
  Mat_<Vec<uchar,3> > smoothed;
  GaussianBlur(image, smoothed, Size(0,0), 0.8);
  WeightedGraph grid = gridGraph(smoothed, CONNECTIVITY_4);
  DisjointSetForest segmentation = felzenszwalbSegment(300, grid, 500);
  cout<<"computing segmentation graph"<<endl;
  LabelledGraph<Mat> segGraph = segmentationGraph<Mat>(smoothed, segmentation, grid);
  cout<<"computing histograms"<<endl;
  colorHistogramLabels(smoothed, segmentation, segGraph, BINS_PER_CHANNEL);
  cout<<"computing kernel"<<endl;
  float kernelValue = 
    treeWalkKernel<Mat>(basisKernel, 2, 2, segGraph, segGraph);
  cout<<"kernel value ="<<kernelValue<<endl;
  assert(false);

  return 0;
}
