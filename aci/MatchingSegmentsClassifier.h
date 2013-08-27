#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <fl/Headers.h>

#include "SegmentAttributes.h"

using namespace cv;
using namespace Eigen;
using namespace std;

class MatchingSegmentClassifier {
private:
	vector<std::tuple<SegmentLabeling, fl::InputVariable*> > features;
	fl::Engine *similarity;
	fl::OutputVariable *similarityOutput;
	bool ignoreFirst;
	// for each training sample, store its class label with
	// the segment labels by features.
	vector<std::tuple<vector<vector<VectorXd> >, int> > trainingLabels;

	void computeSegmentLabels(DisjointSetForest &seg, const Mat_<Vec3b> &image, const Mat_<float> &mask, vector<vector<VectorXd> > &segmentLabels);

	void mostSimilarSegmentLabels(const vector<vector<VectorXd> > &lLabels, const vector<vector<VectorXd> > &sLabels, vector<std::tuple<int, int, double> > &matching, int lNbSeg, int sNbSeg);

public:
	MatchingSegmentClassifier(bool ignoreFirst = false);
	~MatchingSegmentClassifier();

	/**
	* Computes a one to one relation between the segments of two images,
	* where related segments are the most similar in the image. Runs in
	* O(|A||B|) time and O(|A| + |B|) space where A and B are the two
	* segmentations.
	*
	* @param lSeg first segmentation.
	* @param lImage image segmented by lSeg.
	* @param lMask mask of image lImage.
	* @param sSeg first segmentation.
	* @param sImage image segmented by sSeg.
	* @param sMask mask of image sImage.
	* @param labelingFunctions segment labeling functions and associated scale
	* parameters to label segment with, and measure similarity between segments
	* using the gaussian kernel.
	* @param ignoreLast true iff the function must ignore the last segment.
	* Useful when the last segment is the background segment, which is of no
	* interest.
	*/
	vector<std::tuple<int, int, double> > mostSimilarSegments(
		DisjointSetForest &lSeg, const Mat_<Vec3b> &lImage, const Mat_<float> &lMask,
		DisjointSetForest &sSeg, const Mat_<Vec3b> &sImage, const Mat_<float> &sMask);

	/**
	 * Trains the classifier with a given training set.
	 *
	 * @param trainingSet vector of tuples (S, I, M, l) where S is a
	 * segmentation of image I with mask M associated to class label l.
	 */
	void train(vector<std::tuple<DisjointSetForest, Mat_<Vec3b>, Mat_<float>, int> > &trainingSet);

	/**
	 * Predicts the class of an unlabeled sample after training.
	 *
	 * @param segmentation a segmentation of the image to predict the class of.
	 * @param image segmented image to predict the class of.
	 * @param mask mask of the image to predict the class of.
	 * @return the predicted class label of the sample.
	 */
	int predict(DisjointSetForest &segmentation, const Mat_<Vec3b> &image, const Mat_<float> &mask);
};
