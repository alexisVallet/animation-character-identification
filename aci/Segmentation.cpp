#include "Segmentation.h"

#define DEBUG_SEGMENTATION false
#define CONNECTIVITY CONNECTIVITY_4
#define MAX_SEGMENTS 500

static double absoluteDifference(const Mat &m1, const Mat &m2) {
	uchar c1 = m1.at<uchar>(0,0), c2 = m2.at<uchar>(0,0);

	return (double)(c1 > c2 ? c1 - c2 : c2 - c1);
}

static double euclidDistance(const Mat &m1, const Mat &m2) {
	return norm(m1 - m2);
}

static bool compareHueDiff(std::tuple<int,int,double> e1, std::tuple<int,int,double> e2) {
	return get<2>(e1) < get<2>(e2);
}

void fuseByHue(const Mat_<Vec3f> &image, const Mat_<float> &mask, DisjointSetForest &overSegmentation, DisjointSetForest &segmentation) {
	segmentation = overSegmentation;
	vector<VectorXd> averageHues = averageHueLabeling(overSegmentation, image, mask);
	vector<std::tuple<int,int,double> > edges;
	edges.reserve((overSegmentation.getNumberOfComponents() - 1) * overSegmentation.getNumberOfComponents() / 2);

	for (int i = 1; i < overSegmentation.getNumberOfComponents(); i++) {
		for (int j = i + 1; j < overSegmentation.getNumberOfComponents(); j++) {
			edges.push_back(std::tuple<int,int,double>(i,j, abs(averageHues[i](0) - averageHues[j](0))));
		}
	}

	sort(edges.begin(), edges.end(), compareHueDiff);
	vector<int> reverseIndexes(overSegmentation.getNumberOfComponents(), -1);
	map<int,int> rootIndexes = overSegmentation.getRootIndexes();

	for (map<int,int>::iterator it = rootIndexes.begin(); it != rootIndexes.end(); it++) {
		reverseIndexes[(*it).second] = (*it).first;
	}

	for (int i = 0; i < (int)edges.size()/10 ; i++) {
		std::tuple<int,int,double> edge = edges[i];

		segmentation.setUnion(reverseIndexes[get<0>(edge)], reverseIndexes[get<1>(edge)]);
	}
}

void segment(const Mat_<Vec3f> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, int felzenszwalbScale) {
	assert(felzenszwalbScale >= 0);
	WeightedGraph graph = gridGraph(image, CONNECTIVITY_4, mask, euclidDistance, false);
	int minCompSize = countNonZero(mask) / MAX_SEGMENTS;
	DisjointSetForest overSegmentation = felzenszwalbSegment(felzenszwalbScale, graph, minCompSize, mask, VOLUME);
	imshow("segmentation", overSegmentation.toRegionImage(image));
	fuseByHue(image, mask, overSegmentation, segmentation);
	imshow("fused", segmentation.toRegionImage(image));
	waitKey(0);
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

DisjointSetForest segmentationImageToSegmentation(const Mat_<Vec3b> &segmentationImage, const Mat_<float> &mask) {
	assert(segmentationImage.rows == mask.rows && segmentationImage.cols == mask.cols);

	DisjointSetForest segmentation(segmentationImage.rows * segmentationImage.cols + 1);
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

	// fuse small components
	segmentation.fuseSmallComponents(
		gridGraph(
			segmentationImage, 
			CONNECTIVITY_4,
			mask, 
			constOne, 
			false), 
		25, 
		mask);


	// fuse background pixels into one segment
	for (int i = 0; i < segmentation.getNumberOfElements() - 1; i++) {
		pair<int,int> coords = fromRowMajor(mask.cols, i);

		if (mask(coords.first, coords.second) < 0.5) {
			segmentation.setUnion(segmentationImage.rows * segmentationImage.cols, i);
		}
	}

	return segmentation;
}

DisjointSetForest loadSegmentation(Mat_<float> &mask, string segmentationFilename) {
	Mat_<Vec3b> segmentationImage = imread(segmentationFilename);

	return segmentationImageToSegmentation(segmentationImage, mask);
}
