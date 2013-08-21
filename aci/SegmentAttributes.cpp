#pragma once

#include "SegmentAttributes.h"
#define DEBUG_ATTRIBUTES false

vector<VectorXd> averageColorLabeling(DisjointSetForest &segmentation, const Mat_<Vec3b> &image, const Mat_<float> &mask) {
	vector<VectorXd> averageColor;
	averageColor.reserve(segmentation.getNumberOfComponents());

	for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
		VectorXd zeros = VectorXd::Zero(3);
		averageColor.push_back(zeros);
	}
	map<int,int> rootIndexes = segmentation.getRootIndexes();

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (mask(i,j) > 0) {
				int root = segmentation.find(toRowMajor(image.cols, j, i));
				int segmentIndex = rootIndexes[root];
				VectorXd pixColor(3);
				pixColor(0) = image(i,j)[0];
				pixColor(1) = image(i,j)[1];
				pixColor(2) = image(i,j)[2];

				averageColor[segmentIndex] += pixColor / (float)segmentation.getComponentSize(root);
			}
		}
	}

	return averageColor;
}

vector<VectorXd> gravityCenterLabeling(DisjointSetForest &segmentation, const Mat_<Vec3b> &image, const Mat_<float> &mask) {
	vector<Vec2f> centers;
	gravityCenters(image, mask, segmentation, centers);

	vector<VectorXd> eigCenters;
	eigCenters.reserve(centers.size());

	for (int i = 0; i < (int)centers.size(); i++) {
		VectorXd center(2);

		center(0) = centers[i](0) / (double)image.rows;
		center(1) = centers[i](1) / (double)image.cols;

		eigCenters.push_back(center);
	}

	return eigCenters;
}

vector<VectorXd> segmentAreaLabeling(DisjointSetForest &segmentation, const Mat_<Vec3b> &image, const Mat_<float> &mask) {
	vector<VectorXd> areas(segmentation.getNumberOfComponents());
	map<int,int> roots = segmentation.getRootIndexes();

	for (map<int,int>::iterator it = roots.begin(); it != roots.end(); it++) {
		int root = (*it).first;
		VectorXd area(1);
		area(0) = (double)segmentation.getComponentSize(root);

		areas[(*it).second] = area;
	}

	return areas;
}

void pixelsCovarianceMatrixLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, const WeightedGraph &segGraph, LabeledGraph<Matx<float, 3, 1> > &labeledGraph) {
	assert(segmentation.getNumberOfComponents() == segGraph.numberOfVertices());
	vector<Mat> segmentSamples(segmentation.getNumberOfComponents());
	map<int,int> rootIndexes = segmentation.getRootIndexes();

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			Mat coords(1,2,CV_32F);

			coords.at<float>(0,0) = (float)i;
			coords.at<float>(0,1) = (float)j;

			int root = segmentation.find(toRowMajor(image.cols, j, i));
			int segmentIndex = rootIndexes[root];

			if (segmentSamples[segmentIndex].empty()) {
				segmentSamples[segmentIndex] = coords;
			} else {
				Mat tmp;

				vconcat(segmentSamples[segmentIndex], coords, tmp);

				segmentSamples[segmentIndex] = tmp;
			}
		}
	}

	Mat_<Vec3b> regionImage = segmentation.toRegionImage(image);
	labeledGraph = LabeledGraph<Matx<float, 3, 1> >(segGraph.numberOfVertices());
	labeledGraph.copyEdges(segGraph);

	for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
		Mat covarianceMatrix(0, 0, CV_32F);
		Mat mean(0,0,CV_32F);

		calcCovarMatrix(segmentSamples[i], covarianceMatrix, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE, CV_32F);

		Mat eigenvalues;
		Mat eigenvectors;

		eigen(covarianceMatrix, eigenvalues, eigenvectors);

		float axis1 = 2 * sqrt(eigenvalues.at<float>(0,0));
		float axis2 = 2 * sqrt(eigenvalues.at<float>(1,0));
		Mat ev1;
		Mat ev2;
		Mat horizontal(1,2,CV_32F);
		horizontal.at<float>(0,0) = 1;
		horizontal.at<float>(0,1) = 0;
		normalize(eigenvectors.row(0), ev1);
		normalize(eigenvectors.row(1), ev2);
		float angle = (float)acos(horizontal.dot(ev1));

		if (DEBUG_ATTRIBUTES) {
			float degreeAngle = angle * 180 / M_PI;
			cout<<"angle = "<<angle<<" radians and "<<degreeAngle<<" degrees"<<endl;
			ellipse(regionImage, Point((int)mean.at<float>(0,1), (int)mean.at<float>(0,0)), Size((int)axis2, (int)axis1), degreeAngle, 0, 360, Scalar(0,0,255), 2);
		}

		double diagonal = sqrt(pow(image.rows,2.) + pow(image.cols,2.));

		Matx<float, 3, 1> ellipseDescriptor(3, 1);

		ellipseDescriptor(0,0) = axis1 / diagonal;
		ellipseDescriptor(1,0) = axis2 / diagonal;
		ellipseDescriptor(2,0) = angle;

		labeledGraph.addLabel(i, ellipseDescriptor);
	}

	if (DEBUG_ATTRIBUTES) {
		imshow("ellipses", regionImage);
		waitKey(0);
	}
}
