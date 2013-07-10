#include "main.h"

#define TEST true
#define NBCLASSES 12
#define FELZ_SCALE 10000
#define DEPTH 2
#define ARITY 2
#define STATFOLDER "../stats/"

bool compareGraphSize_(const pair<WeightedGraph, int> &p1, const pair<WeightedGraph, int> &p2) {
	return compareGraphSize(p1.first, p2.first);
}

SparseMatrix<double> normalizedSparseLaplacian_(const WeightedGraph &graph, bool bidirectional) {
	VectorXd degrees;

	return normalizedSparseLaplacian(graph, bidirectional, degrees);
}

int main(int argc, char** argv) {
	if (TEST) {
		testPrincipalAnglesClassifier();
	} else {
		cout<<"loading dataset..."<<endl;
		char *charaNames[] = {"rufy", "ray", "miku", "majin", "lupin", "kouji", "jigen", "conan", "chirno", "char", "asuka", "amuro", NULL};
		vector<pair<Mat_<Vec3b>, Mat_<float> > > dataSet;
		Mat_<int> classes;

		loadDataSet("../test/dataset/", charaNames, 5, dataSet, classes);
		
		cout<<"preprocessing..."<<endl;
		vector<pair<Mat_<Vec3b>, Mat_<float> > > processedDataSet;
		processedDataSet.reserve(dataSet.size());

		for (int i = 0; i < (int)dataSet.size(); i++) {
			Mat_<Vec3b> processedImage;
			Mat_<float> processedMask;

			preProcessing(dataSet[i].first, dataSet[i].second, processedImage, processedMask);
			processedDataSet.push_back(pair<Mat_<Vec3b>, Mat_<float> >(processedImage, processedMask));
		}

		cout<<"segmentation..."<<endl;
		vector<pair<WeightedGraph, int> > segmentations;
		segmentations.reserve(dataSet.size());
		CompoundGaussianKernel edgeWeights(5,5,5);

		for (int i = 0; i < (int)dataSet.size(); i++) {
			DisjointSetForest segmentation;
			WeightedGraph segmentationGraph;

			segment(processedDataSet[i].first, processedDataSet[i].second, segmentation, segmentationGraph, FELZ_SCALE);
			WeightedGraph weightedSegGraph = weighEdgesByKernel<float,8,1>(processedDataSet[i].first, processedDataSet[i].second, segmentation, edgeWeights, segmentationGraph);
			segmentations.push_back(pair<WeightedGraph,int>(weightedSegGraph, classes(i,0)));
		}

		cout<<"classification..."<<endl;
		// compute maximum number of vertices
		int maxNbVertices = max_element(segmentations.begin(), segmentations.end(), compareGraphSize_)->first.numberOfVertices();
		KNearestModel statModel(1);
		PrincipalAnglesClassifier classifier(&statModel, normalizedSparseLaplacian_, false, true, maxNbVertices);
		float recognitionRate = classifier.leaveOneOutRecognitionRate(segmentations);

		cout<<"Recognition rate "<<recognitionRate<<endl;
	}
}
