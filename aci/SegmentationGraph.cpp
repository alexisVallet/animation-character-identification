#include "SegmentationGraph.hpp"

Mat_<int> computeBorderLengths(DisjointSetForest &segmentation, WeightedGraph &gridGraph) {
	Mat_<int> borderLengths = Mat_<int>::zeros(segmentation.getNumberOfComponents(), segmentation.getNumberOfComponents());
	map<int,int> rootIndexes = segmentation.getRootIndexes();

	for (int i = 0; i < (int)gridGraph.getEdges().size(); i++) {
		Edge edge = gridGraph.getEdges()[i];
		int srcRoot = segmentation.find(edge.source);
		int dstRoot = segmentation.find(edge.destination);

		if (srcRoot != dstRoot) {
			int src = rootIndexes[srcRoot];
			int dst = rootIndexes[dstRoot];

			borderLengths(src, dst) += 1;
			borderLengths(dst, src) += 1;
		}
	}

	return borderLengths;
}

vector<Vec<float,2> > segmentCenters(Mat_<Vec<uchar,3> > &image, DisjointSetForest &segmentation) {
  int numberOfComponents = segmentation.getNumberOfComponents();
  vector<Vec<float, 2> > centers(numberOfComponents, Vec<int,2>(0,0));
  map<int,int> rootIndexes = segmentation.getRootIndexes();
  
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      int root = segmentation.find(toRowMajor(image.cols, j, i));
      int rootIndex = rootIndexes[root];

      centers[rootIndex] += Vec<float,2>((float)i,(float)j)/((float)segmentation.getComponentSize(root));
    }
  }

  return centers;
}
