#include "SegmentationGraph.hpp"

vector<Vec<float,2> > segmentCenters(Mat_<Vec<uchar,3> > &image, DisjointSetForest &segmentation) {
  int numberOfComponents = segmentation.getNumberOfComponents();
  vector<Vec<float, 2> > centers(numberOfComponents, Vec<int,2>(0,0));
  map<int,int> rootIndexes = segmentation.getRootIndexes();
  
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      int root = segmentation.find(toRowMajor(image.cols, j, i));
      int rootIndex = rootIndexes[root];

      centers[rootIndex] += Vec<float,2>(i,j)/((float)segmentation.getComponentSize(root));
    }
  }

  return centers;
}
