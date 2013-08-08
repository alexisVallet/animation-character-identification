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

static double constOne(const Mat &m1, const Mat &m2) {
	return 1;
}

static bool lexicographicOrder(const Vec3b &v1, const Vec3b &v2) {
	return 
		v1[0] < v2 [0] || 
		(v1[0] == v2[0] &&
		 (v1[1] < v2[1] ||
		  (v1[1] == v2[1] && v1[2] < v2[2])));
}

DisjointSetForest loadSegmentation(Mat_<float> &mask, string segmentationFilename) {
	Mat_<Vec3b> segmentationImage = imread(segmentationFilename);

	assert(segmentationImage.rows == mask.rows && segmentationImage.cols == mask.cols);

	DisjointSetForest segmentation(segmentationImage.rows * segmentationImage.cols);
	typedef bool (*keyComp)(const Vec3b&, const Vec3b&);
	std::map<Vec3b,int, keyComp> colorRepresentant(lexicographicOrder);

	// fuse pixels of the same color
	for (int i = 0; i < segmentationImage.rows; i++) {
		for (int j = 0; j < segmentationImage.cols; j++) {
			if (mask(i,j) > 0.5) {
				Vec3b color = segmentationImage(i,j);
				int linIndex = toRowMajor(segmentationImage.cols, j, i);

				if (colorRepresentant.find(color) == colorRepresentant.end()) {
					colorRepresentant[color] = linIndex;
				} else {
					segmentation.setUnion(colorRepresentant[color], linIndex);
				}
			}
		}
	}

	segmentation.fuseSmallComponents(
		gridGraph(
			segmentationImage, 
			CONNECTIVITY_4,
			mask, 
			constOne, 
			false), 
		25, 
		mask);

	return segmentation;
}
