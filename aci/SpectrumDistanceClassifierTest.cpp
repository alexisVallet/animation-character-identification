#include "SpectrumDistanceClassifierTest.h"

#define TEST false
#define DEBUG false

#define EIG_MU 0.01

using namespace std;
using namespace cv;

void computeRates(
	vector<WeightedGraph> &graphs,
	Mat_<int> &classes,
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
/*
void testSpectrumDistanceClassifier() {
	Mat_<int> classes;
	// loads the dataset
	cout<<"loading dataset"<<endl;
	char *folder = "../test/dataset/";
	char *names[] = {"amuro", "asuka", "char", "chirno", "conan", "jigen", "kouji", "lupin", "majin", "miku", "ray", "rufy"};
	vector<pair<Mat_<Vec<uchar,3> >,Mat_<float> > > dataSet, dataSet2;

	loadDataSet(folder, names, 12, 5, dataSet, classes);

	vector<WeightedGraph> graphs;
	graphs.reserve(dataSet.size());

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
*/