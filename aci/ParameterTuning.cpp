#include "ParameterTuning.h"

#define K_MIN 500
#define K_MAX 5000
#define K_STEP 500

void parameterTuning() {
	cout<<"loading dataset"<<endl;
	// loading dataset
	char *folder = "../test/dataset/";
	char *names[] = {"amuro", "asuka", "char", "chirno", "conan", "jigen", "kouji", "lupin", "majin", "miku", "ray", "rufy"};
	vector<pair<Mat_<Vec3b>, Mat_<float> > > dataSet;
	Mat_<int> classes;

	loadDataSet("../test/dataset/", names, 12, 5, dataSet, classes);

	cout<<"preprocessing"<<endl;
	// pre-processing data set
	vector<pair<Mat_<Vec3b>, Mat_<float> > > preProcessedDataset;
	preProcessedDataset.reserve(dataSet.size());

	for (int i = 0; i < (int)dataSet.size(); i++) {
		Mat_<Vec3b> processedImage;
		Mat_<float> processedMask;

		preProcessing(dataSet[i].first, dataSet[i].second, processedImage, processedMask);
		preProcessedDataset.push_back(pair<Mat_<Vec3b>, Mat_<float> >(processedImage, processedMask));
	}

	for (int k = K_MIN; k < K_MAX; k += K_STEP) {
		// segmentation with various parameters
		vector<DisjointSetForest> segmentations;
		vector<LabeledGraph<Mat> > segGraphs;
		segmentations.reserve(dataSet.size());

		for (int i = 0; i < (int)dataSet.size(); i++) {
			DisjointSetForest segmentation;
			LabeledGraph<Mat> segGraph;

			segment(preProcessedDataset[i].first, preProcessedDataset[i].second, segmentation, segGraph, k);
			segmentations.push_back(segmentation);
			segGraphs.push_back(segGraph);
		}

		double alpha[3] = {5,10};

		// enumerating all combinations of the alpha set
		for (int c = 0; c < 2; c++) {
			for (int x = 0; x < 2; x++) {
				for (int s = 0; s < 2; s++) {
					CompoundGaussianKernel simFunc(alpha[c], alpha[x], alpha[s]);
					vector<WeightedGraph> graphsToClassify;

					graphsToClassify.reserve(dataSet.size());
					for (int i = 0; i < dataSet.size(); i++) {
						graphsToClassify.push_back(weighEdgesByKernel(preProcessedDataset[i].first, preProcessedDataset[i].second, segmentations[i], simFunc, segGraphs[i]));
					}

					KNearestModel nnModel(1);
					SpectrumDistanceClassifier classifier(&nnModel, laplacian);

					float rate = classifier.leaveOneOutRecognitionRate(graphsToClassify, classes);

					cout<<"k = "<<k<<" alphaC = "<<alpha[c]<<" alphaX = "<<alpha[x]<<" alphaS = "<<alpha[s]<<endl;
					cout<<"rate = "<<rate<<endl;
				}
			}
		}
	}
}
