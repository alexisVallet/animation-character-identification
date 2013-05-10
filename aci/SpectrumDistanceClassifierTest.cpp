#include "SpectrumDistanceClassifierTest.h"

#define TEST false
#define DEBUG false
#define CONNECTIVITY CONNECTIVITY_4
#define MAX_SEGMENTS 250
#define GAUSS_SIGMA 0.8
#define MAX_NB_PIXELS 15000
#define FELZENSZWALB_SCALE 1500
#define EIG_MU 0.01

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

LabeledGraph<Mat> groundGraph(const LabeledGraph<Mat> &unGrounded) {
	LabeledGraph<Mat> grounded(unGrounded.numberOfVertices() + 1);

	// copying edges
	for (int i = 0; i < (int)unGrounded.getEdges().size(); i++) {
		Edge edge = unGrounded.getEdges()[i];

		grounded.addEdge(edge.source, edge.destination, edge.weight);
	}

	// assumes all the labels have the same size
	int labelRows = unGrounded.getLabel(0).rows;
	int labelCols = unGrounded.getLabel(0).cols;
	int labelType = unGrounded.getLabel(0).type();

	// adding edges adjacent to the ground vertex, copying labels
	for (int i = 0; i < unGrounded.numberOfVertices(); i++) {
		assert(unGrounded.getLabel(i).rows == labelRows && unGrounded.getLabel(i).cols == labelCols);
		grounded.addLabel(i, unGrounded.getLabel(i));
		grounded.addEdge(i, unGrounded.numberOfVertices(), 1);
	}

	grounded.addLabel(unGrounded.numberOfVertices(), Mat::zeros(labelRows, labelCols, labelType));

	return grounded;
}

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

	resizeImage(smoothed, mask, resized, resizedMask);

	WeightedGraph grid = gridGraph(resized, CONNECTIVITY_4, resizedMask, euclidDistance, false);
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

	if (DEBUG) {
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

void computeRates(
	vector<WeightedGraph> graphs,
	Mat classes,
	vector<pair<string, TrainableStatModel*> > models,
	vector<pair<string, MatrixRepresentation> > representations) {
	for (int i = 0; i < (int)models.size(); i++) {
		cout<<models[i].first<<endl;
		for (int k = 0; k < (int)representations.size(); k++) {
			cout<<representations[k].first<<endl;
			SpectrumDistanceClassifier classifier(models[i].second, representations[k].second, (float)EIG_MU);
			float rate = classifier.leaveOneOutRecognitionRate(graphs, classes);

			cout<<"rate = "<<rate<<endl;;
		}
	}
}

void testSpectrumDistanceClassifier() {
	Mat classes;
	// loads the dataset
	cout<<"loading dataset"<<endl;
	char *folder = "../test/dataset/";
	char *names[] = {"asuka", "amuro", "rufy", "char", "chirno", "majin"};
	vector<pair<Mat_<Vec<uchar,3> >,Mat_<float> > > dataSet, dataSet2;

	vector<WeightedGraph> graphs;

	loadDataSet(folder, names, 6, 5, dataSet, classes);
	loadDataSet(folder, names, 6, 5, dataSet2, classes);

	dataSet.reserve(dataSet.size() * 2);
	
	for (int i = 0; i < (int)dataSet2.size(); i++) {
		dataSet.push_back(dataSet2[i]);
	}

	vconcat(classes, classes, classes);

	cout<<"computing segmentation graphs"<<endl;
	for (int i = 0; i < (int)dataSet.size(); i++) {
		graphs.push_back(computeGraphFrom(dataSet[i].first, dataSet[i].second));
	}

	KNearestModel knnModel;
	BayesModel bayesModel;

	vector<pair<string, TrainableStatModel*> > models;

	models.push_back(pair<string,TrainableStatModel*>("Nearest neighbor", &knnModel));
	//models.push_back(pair<string,TrainableStatModel*>("Bayes", &bayesModel));

	vector<pair<string, MatrixRepresentation> > representations;

	representations.push_back(pair<string, MatrixRepresentation>("Combinatorial Laplacian", laplacian));
	//representations.push_back(pair<string, MatrixRepresentation>("Normalized Laplacian", normalizedLaplacian));

	computeRates(graphs, classes, models, representations);
}