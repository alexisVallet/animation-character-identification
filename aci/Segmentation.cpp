#include "Segmentation.h"

#define DEBUG_SEGMENTATION false
#define CONNECTIVITY CONNECTIVITY_4
#define MAX_SEGMENTS 250

void segment(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, WeightedGraph &segGraph, int felzenszwalbScale) {
	assert(felzenszwalbScale >= 0);
	WeightedGraph grid = gridGraph(image, CONNECTIVITY, mask, euclidDistance, false);
	int minCompSize = countNonZero(mask) / MAX_SEGMENTS;
	segmentation = felzenszwalbSegment(felzenszwalbScale, grid, minCompSize, mask, VOLUME);
	segGraph = segmentationGraph(segmentation, grid);

	if (DEBUG_SEGMENTATION) {
		imshow("image", image);
		waitKey(0);
		Mat regionImage = segmentation.toRegionImage(image);
		segGraph.drawGraph(segmentCenters(image, segmentation), regionImage);
		imshow("segmentation graph", regionImage);
		cout<<"number of components: "<<segmentation.getNumberOfComponents()<<endl;
		waitKey(0);
	}
}
