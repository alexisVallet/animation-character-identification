#include "TreeWalkKernel.hpp"

static int uniformMap(int binsPerChannel, unsigned char channelValue) {
  return floor(((float)channelValue/255.0)*binsPerChannel);
}

void colorHistogramLabels(
    Mat_<Vec<uchar,3> > &image, 
    DisjointSetForest &segmentation, 
    LabelledGraph<Mat> &segmentationGraph,
    int binsPerChannel) {
  int numberOfComponents = segmentation.getNumberOfComponents();
  map<int,int> rootIndexes = segmentation.getRootIndexes();
  vector<Mat> histograms(numberOfComponents);
  const int dims[3] = {binsPerChannel, binsPerChannel, binsPerChannel};

  for (int i = 0; i < numberOfComponents; i++) {
    histograms[i] = Mat(3, dims, CV_32S);
  }

  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      int pixComp =
	rootIndexes[segmentation.find(toRowMajor(image.cols,j,i))];
      Vec<uchar,3> pixColor = image(i,j);
      histograms[pixComp].at<int>(
          uniformMap(binsPerChannel,pixColor[0]),
	  uniformMap(binsPerChannel,pixColor[1]),
	  uniformMap(binsPerChannel,pixColor[2]))++;
    }
  }

  for (int i = 0; i < numberOfComponents; i++) {
    segmentationGraph.addLabel(i, histograms[i]);
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

	if (pi + qi != 0) {
	  float toAdd = pow((float)(pi-qi),2) / (float)(pi + qi);

	  d2 += toAdd;
	}
      }
    }
  }

  float result = lambda * exp(-mu * d2);

  return result;
}

float kroneckerKernel(const Mat &h1, const Mat&h2) {
  Mat diff;

  absdiff(h1, h2, diff);

  return sum(diff)[0] == 0;
}
