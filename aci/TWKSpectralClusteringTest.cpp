#include "TWKSpectralClusteringTest.h"

#define FELZ_SCALE 1000
#define TWK_DEPTH 2
#define TWK_ARITY 2

void testTWKSpectralClustering(ostream &out) {
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

	cout<<"segmentation"<<endl;
	// segmentation
	vector<WeightedGraph> segGraphs;
	vector<DisjointSetForest> segmentations;
	segmentations.reserve(dataSet.size());
	segGraphs.reserve(dataSet.size());

	for (int i = 0; i < (int)dataSet.size(); i++) {
		DisjointSetForest segmentation;
		WeightedGraph segGraph;

		segment(preProcessedDataset[i].first, preProcessedDataset[i].second, segmentation, segGraph, FELZ_SCALE);
		segGraphs.push_back(segGraph);
		segmentations.push_back(segmentation);
	}
	
	cout<<"labeling"<<endl;
	// labeling
	vector<LabeledGraph<Matx<float,8,1> > > labeled;
	CompoundGaussianKernel gaussKernel(5,5,5);

	for (int i = 0; i < (int)dataSet.size(); i++) {	
		LabeledGraph<Matx<float,8,1> > labeledGraph;

		gaussKernel.getLabeling()(preProcessedDataset[i].first, preProcessedDataset[i].second, segmentations[i], segGraphs[i], labeledGraph);

		labeled.push_back(labeledGraph);
	}

	cout<<"embedding"<<endl;
	// embedding
	TWKSpectralClustering<float,8,1> clustering(preProcessedDataset, (MatKernel<float,8,1>*)&gaussKernel, TWK_DEPTH, TWK_ARITY);
	MatrixXd embeddings;

	clustering.embed(segmentations, labeled, 2, embeddings);

	for (int i = 0; i < (int)dataSet.size(); i++) {
		out<<embeddings(i,0)<<", "<<embeddings(i,1)<<endl;
	}
}