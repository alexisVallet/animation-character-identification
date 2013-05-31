#include "TWKSpectralClusteringTest.h"

#define FELZ_SCALE 1000
#define TWK_DEPTH 2
#define TWK_ARITY 2
#define TWK_NBCLASSES 12

void testTWKSpectralClustering(ostream &out) {
	cout<<"loading dataset"<<endl;
	// loading dataset
	char *folder = "../test/dataset/";
	char *names[] = {"amuro", "asuka", "char", "chirno", "conan", "jigen", "kouji", "lupin", "majin", "miku", "ray", "rufy"};
	vector<pair<Mat_<Vec3b>, Mat_<float> > > dataSet;
	Mat_<int> classes;

	loadDataSet("../test/dataset/", names, TWK_NBCLASSES, 5, dataSet, classes);

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
	vector<LabeledGraph<Matx<float,4,1> > > labeled;
	TWBasisKernel twKernel(5, 0);

	for (int i = 0; i < (int)dataSet.size(); i++) {	
		LabeledGraph<Matx<float,4,1> > labeledGraph;

		twKernel.getLabeling()(preProcessedDataset[i].first, preProcessedDataset[i].second, segmentations[i], segGraphs[i], labeledGraph);

		labeled.push_back(labeledGraph);
	}

	cout<<"clustering"<<endl;
	// embedding
	TWKSpectralClustering<float,4,1> clustering(preProcessedDataset, (MatKernel<float,4,1>*)&twKernel, TWK_DEPTH, TWK_ARITY);
	VectorXi classLabels;
	
	clustering.cluster(segmentations, labeled, TWK_NBCLASSES, classLabels);
	cout<<classLabels<<endl;
}
