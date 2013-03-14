#include "TreeWalkKernel.hpp"

void colorHistogramLabels(
    Mat_<Vec<uchar,3> > &image, 
    DisjointSetForest &segmentation, 
    LabelledGraph<Mat> &segmentationGraph,
    int binsPerChannel) {
  int numberOfComponents = segmentation.getNumberOfComponents();
  map<int,int> rootIndexes = segmentation.getRootIndexes();
  vector<Mat_<uchar> > masks(
    numberOfComponents, 
    Mat::zeros(image.rows,image.cols,CV_8U));

  // computes mask for each component
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      int pixComp = 
	rootIndexes[segmentation.find(toRowMajor(image.cols,j,i))];

      masks[pixComp][i][j] = 1;
    }
  }

  int histSizes[3] = {binsPerChannel, binsPerChannel, binsPerChannel};
  float rRange[2] = {0, 256};
  const float *ranges[3] = {rRange, rRange, rRange};
  int channels[3] = {0, 1, 2};

  // compute the histogram for each component and add it to 
  // the corresponding vertex
  for (int i = 0; i < numberOfComponents; i++) {
    Mat histogram(3, histSizes, CV_32S);

    calcHist(&image, 1, channels, masks[i], histogram, 3, histSizes, ranges, true, false);
    segmentationGraph.addLabel(i, histogram);
  }
}

void medianColorLabels(
  Mat_<Vec<uchar,3> > &image,
  DisjointSetForest &segmentation,
  LabelledGraph<Vec<uchar,3> > &segmentationGraph) {
  vector<Vec<float,3> > medianColors(segmentation.getNumberOfComponents(), Vec<float,3>(0,0,0));
  map<int,int> rootIndexes = segmentation.getRootIndexes();

  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      int comp = rootIndexes[segmentation.find(toRowMajor(image.cols, j, i))];
      medianColors[comp] += Vec<float,3>(image(i,j)[0],image(i,j)[1],image(i,j)[2]) / segmentation.getComponentSize(comp);
    }
  }

  for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
    Vec<float,3> color = medianColors[i];

    segmentationGraph.addLabel(i, Vec<uchar,3>(color[0],color[1],color[2]));
  }
}

float khi2Kernel(int binsPerChannel, float lambda, float mu, const Mat &h1, const Mat &h2) {
  float d2 = 0;

  for (int r = 0; r < binsPerChannel; r++) {
    for (int g = 0; g < binsPerChannel; g++) {
      for (int b = 0; b < binsPerChannel; b++) {
	int pi = h1.at<int>(r,g,b);
	int qi = h2.at<int>(r,g,b);

	d2 += pow(pi-qi,2) / (pi + qi);
      }
    }
  }

  return lambda * exp(-mu * d2);
}
