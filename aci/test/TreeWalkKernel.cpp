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
  Mat_<Vec<uchar,3> > imageRGB = imread(ACI_SOURCE_DIR "/aci/test/mickey.png");
  Mat_<Vec<uchar,3> > image;
  cvtColor(imageRGB, image, CV_RGB2Lab);
  Mat_<Vec<uchar,3> > smoothed;
  GaussianBlur(image, smoothed, Size(0,0), 0.8);
  WeightedGraph grid = gridGraph(smoothed, CONNECTIVITY_4);
  DisjointSetForest segmentation = felzenszwalbSegment(300, grid, 500);
  Mat_<Vec<uchar,3> > segmentationImage = segmentation.toRegionImage(image);
  cout<<"computing segmentation graph"<<endl;
  LabelledGraph<Mat> segGraph = segmentationGraph<Mat>(smoothed, segmentation, grid);
  segGraph.drawGraph(segmentCenters(image,segmentation), segmentationImage);
  imshow("segmentation graph", segmentationImage);
  waitKey(0);
  cout<<"computing histograms"<<endl;
  colorHistogramLabels(smoothed, segmentation, segGraph, BINS_PER_CHANNEL);
  cout<<"computing kernel"<<endl;
  float kernelValue = 
    treeWalkKernel<Mat>(kroneckerKernel, 10, 10, segGraph, segGraph);
  cout<<"kernel value = "<<kernelValue<<endl;

  return 0;
}
