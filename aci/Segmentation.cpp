#include "Segmentation.h"

#define DEBUG_SEGMENTATION false
#define CONNECTIVITY CONNECTIVITY_4
#define MAX_SEGMENTS 250
#define MAX_NB_PIXELS 15000
#define FELZENSZWALB_SCALE 1500

WeightedGraph computeGraphFrom(Mat_<Vec<uchar,3> > &bgrImage, Mat_<float> &mask) {
	cout<<"equalizing color histogram"<<endl;
	// equalize the color histogram
	Mat_<Vec3b> equalized;
	equalizeColorHistogram(bgrImage, mask, equalized);
	cout<<"filtering outlines"<<endl;
	// filter out the outlines
	Mat_<Vec<uchar,3> > smoothedRgb;

	KuwaharaFilter(equalized, smoothedRgb, 11);

	cout<<"converting to Lab"<<endl;
	// convert to Lab and resize
	Mat_<Vec3b> smoothed;
	cvtColor(smoothedRgb, smoothed, CV_RGB2Lab);

	cout<<"resizing"<<endl;
	Mat_<Vec3b> resized;
	Mat_<float> resizedMask;

	resizeImage(smoothed, mask, resized, resizedMask, MAX_NB_PIXELS);

	WeightedGraph grid = gridGraph(resized, CONNECTIVITY, resizedMask, euclidDistance, false);
	int minCompSize = countNonZero(resizedMask) / MAX_SEGMENTS;
	DisjointSetForest segmentation = felzenszwalbSegment(FELZENSZWALB_SCALE, grid, minCompSize, resizedMask);
	LabeledGraph<Mat> segGraph = segmentationGraph<Mat>(resized, segmentation, grid);
	
	vector<Labeling> labelings;

	labelings.push_back(gravityCenterLabels);
	labelings.push_back(averageColorLabels);
	labelings.push_back(segmentIndexLabels);
	labelings.push_back(pixelsCovarianceMatrixLabels);

	concatenateLabelings(labelings, resized, resizedMask, segmentation, segGraph);

	// adding a "ground" vertex labelled with the 0 vector and adjacent to all the
	// vertices in the graph.
	LabeledGraph<Mat> groundedGraph = groundGraph(segGraph);

	CompoundGaussianKernel similarityFunctor(computeBorderLengths(segmentation, grid));
	WeightedGraph finalGraph = weighEdgesByKernel(similarityFunctor, groundedGraph);

	if (DEBUG_SEGMENTATION) {
		imshow("equalized", equalized);
		showHistograms(smoothed, mask, 255);
		imshow("filtered", smoothed);
		waitKey(0);
		Mat regionImage = segmentation.toRegionImage(resized);
		//segGraph.drawGraph(segmentCenters(smoothed, segmentation), regionImage);
		imshow("segmentation graph", regionImage);
		cout<<"number of components: "<<segmentation.getNumberOfComponents()<<endl;
		waitKey(0);
	}

	return finalGraph;
}