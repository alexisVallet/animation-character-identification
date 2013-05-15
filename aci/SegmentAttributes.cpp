#pragma once

#include "SegmentAttributes.h"
#define DEBUG_ATTRIBUTES false

static int uniformMap(int binsPerChannel, unsigned char channelValue) {
	return (int)floor(((float)channelValue/255.0)*(binsPerChannel-1));
}

void colorHistogramLabels(
	Mat_<Vec<uchar,3> > &image,
	DisjointSetForest &segmentation,
	LabeledGraph<Mat> &segmentationGraph,
	int binsPerChannel) {
		int numberOfComponents = segmentation.getNumberOfComponents();
		map<int,int> rootIndexes = segmentation.getRootIndexes();
		vector<Mat> histograms(numberOfComponents);
		int dims[3] = {binsPerChannel, binsPerChannel, binsPerChannel};
		for (int i = 0; i < numberOfComponents; i++) {
			histograms[i] = Mat(3, dims, CV_32S);

			for (int j = 0; j < dims[0]; j++) {
				for (int k = 0; k < dims[1]; k++) {
					for (int l = 0; l < dims[2]; l++) {
						histograms[i].at<int>(j,k,l) = 0;
					}
				}
			}
		}

		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				int pixComp =
					rootIndexes[segmentation.find(toRowMajor(image.cols,j,i))];
				Vec<uchar,3> pixColor = image(i,j);
				int r = uniformMap(binsPerChannel,pixColor[0]);
				int g = uniformMap(binsPerChannel,pixColor[1]);
				int b = uniformMap(binsPerChannel,pixColor[2]);
				histograms[pixComp].at<int>(r, g, b)++;
			}
		}

		for (int i = 0; i < numberOfComponents; i++) {
			segmentationGraph.addLabel(i, histograms[i]);
		}
}

void averageColorLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, const WeightedGraph &segGraph, LabeledGraph<Matx<float,3,1> > &labeledGraph) {
	assert(image.rows == mask.rows && image.cols == mask.cols);
	assert(segmentation.getNumberOfComponents() == segGraph.numberOfVertices());
	vector<Vec3f> averageColor;
	averageColor.reserve(segmentation.getNumberOfComponents());

	for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
		averageColor.push_back(Vec3f(0,0,0));
	}
	map<int,int> rootIndexes = segmentation.getRootIndexes();

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (mask(i,j) > 0) {
				int root = segmentation.find(toRowMajor(image.cols, j, i));
				int segmentIndex = rootIndexes[root];
				Vec3f pixColor = Vec3f(image(i,j));

				averageColor[segmentIndex] += pixColor / (float)segmentation.getComponentSize(root);
			}
		}
	}

	labeledGraph = LabeledGraph<Matx<float, 3, 1> >(segGraph.numberOfVertices());

	labeledGraph.copyEdges(segGraph);

	for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
		labeledGraph.addLabel(i, averageColor[i]/255);
	}
}

void gravityCenterLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, const WeightedGraph &segGraph, LabeledGraph<Matx<float, 2, 1> > &labeledGraph) {
	assert(image.rows == mask.rows && image.cols == mask.cols);
	assert(segmentation.getNumberOfComponents() == segGraph.numberOfVertices());
	
	vector<Vec2f> gravityCenters;
	gravityCenters.reserve(segmentation.getNumberOfComponents());

	for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
		gravityCenters.push_back(Vec2f(0,0));
	}
	map<int,int> rootIndexes = segmentation.getRootIndexes();

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (mask(i,j) > 0) {
				int root = segmentation.find(toRowMajor(image.cols, j, i));
				int segmentIndex = rootIndexes[root];
				Vec2f position = Vec2f((float)i,(float)j);

				gravityCenters[segmentIndex] += position / (float)segmentation.getComponentSize(root);
			}
		}
	}

	labeledGraph = LabeledGraph<Matx<float,2,1> >(segGraph.numberOfVertices());

	labeledGraph.copyEdges(segGraph);

	for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
		Matx<float, 2, 1> gravityCenter(2,1);

		gravityCenter(0,0) = gravityCenters[i](0) / image.rows;
		gravityCenter(1,0) = gravityCenters[i](1) / image.cols;

		labeledGraph.addLabel(i, gravityCenter);
	}
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
