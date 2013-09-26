/** @file */
#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "MatchingSegmentsClassifier.h"

using namespace std;
using namespace Eigen;
using namespace cv;

class PaletteProjectionClassifier {
public:
	typedef std::tuple<DisjointSetForest, Mat_<Vec3f>, Mat_<float>, int> TrainingSample;
	// l, a, b, occurences
	typedef std::tuple<int,int,int,int> HistEntry;

private:
	float lrange[2];
	float arange[2];
	float brange[2];
	double magnification;
	double suppression;
	int nbBins;
	MatchingSegmentClassifier internalClassifier;
	MatrixXd projection;
	vector<int> labels;
	
	int get1DBin(float x, float range[]);

public:
	PaletteProjectionClassifier(int nbBins = 5, double magnification = 10, double suppression = 0.1);

	void computePalette(const Mat_<Vec3f> &image, const Mat_<float> &mask, VectorXd &palette, vector<HistEntry> &flatHistogram);

	void paletteRewrite(const Mat_<Vec3f> &image, const Mat_<float> &mask, const VectorXd &newPalette, const vector<HistEntry> &sortedHistogram, Mat_<Vec3f> &repaletted);

	void train(vector<TrainingSample> &samples);

	int predict(DisjointSetForest &segmentation, const Mat_<Vec3f> &image, const Mat_<float> &mask);
};
