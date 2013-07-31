#include "DirichletEigenSubspaceClassifierTest.h"

void testDirichletEigenSubspaceClassifier() {
	cout<<"loading dataset..."<<endl;
	char *charaNames[] = {"rufy", "ray", "miku", "majin", "lupin", "kouji", "jigen", "conan", "chirno", "char", "asuka", "amuro", NULL};
	vector<std::tuple<Mat_<Vec3b>, Mat_<float> > > dataSet;
	Mat_<int> classes;

	loadDataSet("../test/dataset/", charaNames, 5, dataSet, classes);
	vector<pair<string,Vector2d>, aligned_allocator<pair<string,Vector2d> > > facePositions = loadFacePositions("../test/dataset/faceData.csv");

	cout<<"preprocessing..."<<endl;
	vector<pair<Mat_<Vec3b>, Mat_<float> > > processedDataSet;
	processedDataSet.reserve(dataSet.size());

	for (int i = 0; i < (int)dataSet.size(); i++) {
		Mat_<Vec3b> processedImage;
		Mat_<float> processedMask;

		preProcessing(get<0>(dataSet[i]), get<1>(dataSet[i]), processedImage, processedMask);
		processedDataSet.push_back(pair<Mat_<Vec3b>, Mat_<float> >(processedImage, processedMask));
	}

	cout<<"segmentation..."<<endl;
	vector<pair<WeightedGraph, int> > segmentations;
	segmentations.reserve(dataSet.size());
	CompoundGaussianKernel edgeWeights(5,5,5);

	for (int i = 0; i < (int)dataSet.size(); i++) {
		DisjointSetForest segmentation;
		WeightedGraph segmentationGraph;

		segment(processedDataSet[i].first, processedDataSet[i].second, segmentation, segmentationGraph);
		WeightedGraph weightedSegGraph = weighEdgesByKernel<float,8,1>(processedDataSet[i].first, processedDataSet[i].second, segmentation, edgeWeights, segmentationGraph);
		segmentations.push_back(pair<WeightedGraph,int>(weightedSegGraph, classes(i,0)));
	}

	cout<<"computing face vertices..."<<endl;
	for (int i = 0; i < (int)dataSet.size(); i++) {
		// resize face coordinates to take image resizing into account

	}
}
