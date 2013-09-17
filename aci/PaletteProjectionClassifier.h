#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "MatchingSegmentsClassifier.h"

using namespace std;
using namespace Eigen;
using namespace cv;

class PaletteProjectionClassifier {
private:
	double magnification;
	double suppression;
	double sigma;
	int nbBins;
	MatchingSegmentClassifier internalClassifier;
	MatrixXd projection;
	vector<int> labels;
	typedef std::tuple<DisjointSetForest, Mat_<Vec3f>, Mat_<float>, int> TrainingSample;

	void computePalette(const Mat_<Vec3f> &image, const Mat_<float> &mask, VectorXd &palette);

public:
	PaletteProjectionClassifier(int nbBins = 5, double magnification = 10, double suppression = 0.1, double sigma = 1);

	void train(vector<TrainingSample> &samples);
};
