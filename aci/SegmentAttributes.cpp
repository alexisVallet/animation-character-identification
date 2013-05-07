#pragma once

#include "SegmentAttributes.h"

static int uniformMap(int binsPerChannel, unsigned char channelValue) {
	return floor(((float)channelValue/255.0)*(binsPerChannel-1));
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

void averageColorLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph) {
	assert(image.rows == mask.rows && image.cols == mask.cols);
	vector<Vec3f> averageColor;
	averageColor.reserve(segmentation.getNumberOfComponents());

	for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
		averageColor.push_back(Vec3f(0,0,0));
	}
	map<int,int> rootIndexes = segmentation.getRootIndexes();

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (mask(i,j) > 0) {
				int root = rootIndexes[segmentation.find(toRowMajor(image.cols, j, i))];
				Vec3f pixColor = Vec3f(image(i,j));

				averageColor[root] += pixColor / (float)segmentation.getComponentSize(root);
			}
		}
	}

	for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
		segmentationGraph.addLabel(i, Mat(averageColor[i]));
	}
}

void gravityCenterLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph) {
	assert(image.rows == mask.rows && image.cols == mask.cols);
	vector<Vec2f> gravityCenters;
	gravityCenters.reserve(segmentation.getNumberOfComponents());

	for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
		gravityCenters.push_back(Vec2f(0,0));
	}
	map<int,int> rootIndexes = segmentation.getRootIndexes();

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (mask(i,j) > 0) {
				int root = rootIndexes[segmentation.find(toRowMajor(image.cols, j, i))];
				Vec2f pixColor = Vec2f(i,j);

				gravityCenters[root] += pixColor / (float)segmentation.getComponentSize(root);
			}
		}
	}

	for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
		segmentationGraph.addLabel(i, Mat(gravityCenters[i]));
	}
}

void concatenateLabelings(const vector<Labeling> &labelings, const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph) {
	vector<Mat> labels(segmentationGraph.numberOfVertices());

	for (int i = 0; i < labelings.size(); i++) {
		cout<<"computing labelin "<<i<<endl;
		LabeledGraph<Mat> copy = segmentationGraph;

		labelings[i](image, mask, segmentation, copy);

		for (int j = 0; j < segmentationGraph.numberOfVertices(); j++) {
			if (labels[j].empty()) {
				labels[j] = copy.getLabel(j);
			} else {
				Mat tmp;

				cout<<"labels["<<j<<"]: ("<<labels[j].rows<<","<<labels[j].cols<<")"<<endl;
				cout<<"new label: ("<<copy.getLabel(j).rows<<","<<copy.getLabel(j).cols<<")"<<endl;
				vconcat(labels[j], copy.getLabel(j), tmp);

				cout<<"concatenation computed"<<endl;
				labels[j] = tmp;
			}
		}
	}

	for (int i = 0; i < segmentationGraph.numberOfVertices(); i++) {
		segmentationGraph.addLabel(i, labels[i]);
	}
}

void pixelsCovarianceMatrixLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph) {
	assert(segmentation.getNumberOfComponents() == segmentationGraph.numberOfVertices());
	vector<Mat> segmentSamples(segmentation.getNumberOfComponents());
	map<int,int> rootIndexes = segmentation.getRootIndexes();

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			Mat coords(1,2,CV_32F);

			coords.at<float>(0,0) = i;
			coords.at<float>(0,1) = j;

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

	for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
		Mat covarianceMatrix(0, 0, CV_32F);
		Mat mean(0,0,CV_32F);

		calcCovarMatrix(segmentSamples[i], covarianceMatrix, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32F);

		Mat covarianceVector = covarianceMatrix.reshape(0, 4);

		segmentationGraph.addLabel(i, covarianceVector);
	}
}