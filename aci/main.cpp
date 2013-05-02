#include "main.h"
#include "KuwaharaFilter.h"

#define TEST false
#define DEBUG false
#define BLUR_SIGMA 0.8
#define CONNECTIVITY CONNECTIVITY_4
#define MAX_SEGMENTS 50
#define BINS_PER_CHANNEL 16
#define EIG_MU 1
#define GAUSS_SIGMA 100
#define KHI_MU 0.01
#define KHI_LAMBDA 0.75
#define MAX_NB_PIXELS 15000

using namespace std;
using namespace cv;

void resizeImage(const Mat_<Vec<uchar,3> > &image, const Mat_<float> &mask, Mat_<Vec<uchar,3> > &resizedImage, Mat_<float> &resizedMask) {
	int nbPixels = countNonZero(mask);

	if (nbPixels > MAX_NB_PIXELS) {
		double ratio = sqrt((double)MAX_NB_PIXELS / (double)nbPixels);

		resize(image, resizedImage, Size(), ratio, ratio);
		resize(mask, resizedMask, Size(), ratio, ratio, INTER_NEAREST);
	} else {
		resizedImage = image;
		resizedMask = mask;
	}
}

LabeledGraph<Mat> computeGraphFrom(Mat_<Vec<uchar,3> > &rgbImage, Mat_<float> &mask) {
	// filter the image for better segmentation
	Mat_<Vec<uchar,3> > smoothedRgb;

	KuwaharaFilter(rgbImage, smoothedRgb, 11);
	Mat_<Vec3b> smoothed;

	cvtColor(smoothedRgb, smoothed, CV_RGB2Lab);
	Mat_<Vec3b> resized;
	Mat_<float> resizedMask;

	resizeImage(smoothed, mask, resized, resizedMask);

	WeightedGraph graph = kNearestGraph(resized, resizedMask, 4, euclidDistance, false);
	WeightedGraph grid = gridGraph(resized, CONNECTIVITY_4, resizedMask, euclidDistance, false);
	int minCompSize = countNonZero(resizedMask) / MAX_SEGMENTS;
	DisjointSetForest segmentation = felzenszwalbSegment(100, graph, minCompSize, resizedMask);
	LabeledGraph<Mat> segGraph = segmentationGraph<Mat>(
		resized,
		segmentation,
		graph);
	cout<<"computing average color labels"<<endl;
	
	vector<Labeling> labelings;

	labelings.push_back(gravityCenterLabels);
	labelings.push_back(averageColorLabels);

	concatenateLabelings(labelings, resized, resizedMask, segmentation, segGraph);

	if (DEBUG) {
		showHistograms(smoothed, mask, 255);
		imshow("filtered", smoothed);
		waitKey(0);
		Mat regionImage = segmentation.toRegionImage(resized);
		//segGraph.drawGraph(segmentCenters(smoothed, segmentation), regionImage);
		imshow("segmentation graph", regionImage);
		waitKey(0);
	}

	return segGraph;
}

static double gaussianKernel_(const Mat &h1, const Mat &h2) {
	const double sigmaC = 0.5;
	const double sigmaX = 0.8;

	return gaussianKernel(sigmaC, h1.rowRange(0,3), h2.rowRange(0,3)) * gaussianKernel(sigmaX, h1.rowRange(3,5), h2.rowRange(3,5));
}

void computeRates(
	vector<LabeledGraph<Mat> > graphs,
	Mat classes,
	vector<pair<string, TrainableStatModel*> > models, 
	vector<pair<string, MatKernel> > kernels, 
	vector<pair<string, MatrixRepresentation> > representations) {
	for (int i = 0; i < (int)models.size(); i++) {
		cout<<models[i].first<<endl;
		for (int j = 0; j < (int)kernels.size(); j++) {
			cout<<kernels[j].first<<endl;
			for (int k = 0; k < (int)representations.size(); k++) {
				cout<<representations[k].first<<endl;
				SpectrumDistanceClassifier classifier(kernels[j].second, models[i].second, representations[k].second, EIG_MU);
				float rate = classifier.leaveOneOutRecognitionRate(graphs, classes);

				cout<<"rate = "<<rate<<endl;;
			}
		}
	}
}

int main(int argc, char** argv) {
	if (TEST) {
		
		Mat classes;

		// loads the dataset
		cout<<"loading dataset"<<endl;
		char *folder = "../test/dataset/";
		char *names[] = {"amuro", "asuka", "char", "chirno", "conan", "jigen", "kouji", "lupin", "majin", "miku", "ray", "rufy"};
		vector<pair<Mat_<Vec<uchar,3> >,Mat_<float> > > dataSet;
		vector<LabeledGraph<Mat> > graphs;
		
		loadDataSet(folder, names, 12, 5, dataSet, classes);

		Mat_<Vec3b> dest;

		KuwaharaFilter(dataSet[3].first,dest,5);

		imshow("window", dest);
		waitKey(0);

	} else {
		Mat classes;
		// loads the dataset
		cout<<"loading dataset"<<endl;
		char *folder = "../test/dataset/";
		char *names[] = {"amuro", "asuka", "char", "chirno", "conan", "jigen", "kouji", "lupin", "majin", "miku", "ray", "rufy"};
		vector<pair<Mat_<Vec<uchar,3> >,Mat_<float> > > dataSet;
		vector<LabeledGraph<Mat> > graphs;
		
		loadDataSet(folder, names, 12, 5, dataSet, classes);

		cout<<"computing segmentation graphs"<<endl;
		for (int i = 0; i < dataSet.size(); i++) {
			graphs.push_back(computeGraphFrom(dataSet[i].first, dataSet[i].second));
		}
		
		KNearestModel knnModel;
		BayesModel bayesModel;

		vector<pair<string, TrainableStatModel*> > models;

		models.push_back(pair<string,TrainableStatModel*>("Nearest neighbor", &knnModel));
		//models.push_back(pair<string,TrainableStatModel*>("Bayes", &bayesModel));
	
		vector<pair<string, MatKernel> > kernels;

		//kernels.push_back(pair<string, MatKernel>("Dot product", dotProductKernel));
		kernels.push_back(pair<string, MatKernel>("Gaussian kernel", gaussianKernel_));
		//kernels.push_back(pair<string, MatKernel>("Khi2 kernel", khi2Kernel_));

		vector<pair<string, MatrixRepresentation> > representations;

		representations.push_back(pair<string, MatrixRepresentation>("Combinatorial Laplacian", laplacian));
		//representations.push_back(pair<string, MatrixRepresentation>("Normalized Laplacian", normalizedLaplacian));

		computeRates(graphs, classes, models, kernels, representations);
	}

	return 0;
}
