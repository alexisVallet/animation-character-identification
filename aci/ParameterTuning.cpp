#include "ParameterTuning.h"

#define K_MIN 500
#define K_MAX 5000
#define K_STEP 500
#define L_MIN 1


void runClassificationFor(
	ostream &outStream,
	const vector<pair<Mat_<Vec3b>,Mat_<float> > > &dataSet,
	const Mat_<int> &classes,
	const vector<int> felzScales, 
	const vector<double> alphaCs,
	const vector<double> alphaXs,
	const vector<double> alphaSs,
	int nbEigValSteps) {
	for (int fki = 0; fki < (int)felzScales.size(); fki++) {
		int fk = felzScales[fki];
		// segmentation with various parameters
		vector<DisjointSetForest> segmentations;
		vector<WeightedGraph> segGraphs;
		segmentations.reserve(dataSet.size());

		for (int i = 0; i < (int)dataSet.size(); i++) {
			DisjointSetForest segmentation;
			WeightedGraph segGraph;

			segment(dataSet[i].first, dataSet[i].second, segmentation, segGraph, fk);
			segmentations.push_back(segmentation);
			segGraphs.push_back(segGraph);
		}

		int largestNbVertices = max_element(segGraphs.begin(), segGraphs.end(), compareGraphSize)->numberOfVertices();
		int step = (largestNbVertices - 1) / nbEigValSteps;

		step = step == 0 ? 1 : step;

		// enumerating all combinations of the alpha set
		for (int ci = 0; ci < (int)alphaCs.size(); ci++) {
			double alphaC = alphaCs[ci];
			for (int xi = 0; xi < (int)alphaXs.size(); xi++) {
				double alphaX = alphaXs[xi];
				for (int si = 0; si < (int)alphaSs.size(); si++) {
					double alphaS = alphaSs[si];
					for (int l = 1; l < largestNbVertices; l += step) {
						CompoundGaussianKernel simFunc(alphaC, alphaX, alphaS);
						vector<WeightedGraph> graphsToClassify;

						graphsToClassify.reserve(dataSet.size());
						for (int i = 0; i < (int)dataSet.size(); i++) {
							graphsToClassify.push_back(weighEdgesByKernel<float,8,1>(dataSet[i].first, dataSet[i].second, segmentations[i], simFunc, segGraphs[i]));
						}

						KNearestModel nnModel(1);
						SpectrumDistanceClassifier classifier(&nnModel, laplacian, l);

						float rate = classifier.leaveOneOutRecognitionRate(graphsToClassify, classes);
						float eigenvaluesFraction = (float)l/(float)largestNbVertices;

						cout<<fk<<", "<<alphaC<<", "<<alphaX<<", "<<alphaS<<", "<<eigenvaluesFraction<<", "<<rate<<endl;
					}
				}
			}
		}
	}
}

void parameterTuning(ostream &outStream) {
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

	vector<int> felzScale;
	vector<double> alphaX, alphaC, alphaS;

	for (int fk = K_MIN; fk < K_MAX; fk += K_STEP) {
		felzScale.push_back(fk);
	}

	alphaX.push_back(5);
	alphaC.push_back(5);
	alphaS.push_back(5);
	int nbEigValSteps = 10;

	runClassificationFor(
		outStream,
		preProcessedDataset,
		classes,
		felzScale,
		alphaC,
		alphaX,
		alphaS,
		nbEigValSteps);
}
