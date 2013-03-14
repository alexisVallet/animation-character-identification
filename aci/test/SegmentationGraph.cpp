#include <opencv2/opencv.hpp>
#include <boost/graph/boyer_myrvold_planar_test.hpp>

#include "SegmentationGraph.hpp"
#include "Felzenszwalb.hpp"
#include "../ACIConfig.h"

int main() {
  Mat_<Vec<uchar,3> > image = imread(ACI_SOURCE_DIR "/aci/test/mickey.png");
  Mat_<Vec<uchar,3> > smoothed;
  GaussianBlur(image, smoothed, Size(0,0), 0.8);
  WeightedGraph grid = gridGraph(smoothed, CONNECTIVITY_4);
  DisjointSetForest segmentation = felzenszwalbSegment(150, grid, 50);
  LabelledGraph<Mat> segGraph = segmentationGraph<Mat>(smoothed, segmentation, grid);
  Mat_<Vec<uchar,3> > segmentationImage = segmentation.toRegionImage(image);
  adjacency_list<vecS,vecS,bidirectionalS,property<vertex_index_t, int> > 
    boostGraph = segGraph.toBoostGraph();
  bool isPlanar = boyer_myrvold_planarity_test(boostGraph);
  cout<<"Planarity: "<<isPlanar<<endl;
  assert(isPlanar);
  segGraph.drawGraph(segmentCenters(image,segmentation), segmentationImage);
  imshow("segmentation graph", segmentationImage);
  waitKey(0);

  return 0;
}
