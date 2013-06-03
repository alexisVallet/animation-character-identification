#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "LabeledGraph.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

/**
 * Abstract class for segmentation graph clustering methods.
 */
template < typename _Tp, int m, int n>
class SegmentationGraphClustering {
public:
	/**
	 * Clusters segmentation graphs of images by some method.
	 *
	 * @param segmentations segmentations of images corresponding to the graphs.
	 * @param samples segmentation graphs to cluster.
	 * @param nbClasses number of classes to cluster the graphs into.
	 * @param classLables class label for each graph, in the {0, 1, ..., nbClasses - 1} set.
	 */
	virtual void cluster(vector<DisjointSetForest> &segmentations, const vector<LabeledGraph<Matx<_Tp, m, n> > > &samples, int nbClasses, VectorXi &classLabels) = 0;
};
