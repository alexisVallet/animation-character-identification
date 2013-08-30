#include "main.h"

#define DEBUG true
#define STATFOLDER "../stats/"
#define COLOR_SIGMA 25
#define CENTERS_SIGMA 1
#define AREA_SIGMA 250
#define NB_EIGENVECTORS 7

int main(int argc, char** argv) {
	char *charaNames[] = {"rufy", "ray", NULL};
	vector<std::tuple<Mat_<Vec3b>, Mat_<float> > > dataset;
	Mat_<int> classes;
	cout<<"loading dataset..."<<endl;
	loadDataSet("../test/dataset/", charaNames, 5, dataset, classes);

	cout<<"preprocessing"<<endl;
	vector<std::tuple<Mat_<Vec3f>, Mat_<float> > > processedDataset;
	processedDataset.reserve(dataset.size());

	for (int i = 0; i < dataset.size(); i++) {
		std::tuple<Mat_<Vec3f>, Mat_<float> > processedSample;

		preProcessing(get<0>(dataset[i]), get<1>(dataset[i]), get<0>(processedSample), get<1>(processedSample));

		processedDataset.push_back(processedSample);
	}

	cout<<"segmentation"<<endl;
	vector<DisjointSetForest> segmentations;
	segmentations.reserve(processedDataset.size());

	for (int i = 0; i < (int)processedDataset.size(); i++) {
		DisjointSetForest segmentation;

		segment(get<0>(processedDataset[i]), get<1>(processedDataset[i]), segmentation);

		segmentations.push_back(segmentation);
	}

	cout<<"classification"<<endl;

	float rate = 0;

	for (int i = 0; i < (int)processedDataset.size(); i++) {
		typedef std::tuple<DisjointSetForest, Mat_<Vec3f>, Mat_<float>, int > TrainingSample;
		MatchingSegmentClassifier classifier(true);
		vector<TrainingSample> trainingSet;
		trainingSet.reserve(processedDataset.size() - 1);

		for (int j = 0; j < (int)processedDataset.size(); j++) {
			if (i != j) {
				trainingSet.push_back(TrainingSample(segmentations[j], get<0>(processedDataset[j]), get<1>(processedDataset[j]), classes(j,0)));
			}
		}

		classifier.train(trainingSet);
		int actual = classifier.predict(segmentations[i], get<0>(processedDataset[i]), get<1>(processedDataset[i]));

		cout<<"predicted sample "<<i<<" in class "<<actual<<", expected "<<classes(i,0)<<endl;

		if (actual == classes(i,0)) {
			rate++;
		}
	}

	rate = rate / (float)processedDataset.size();

	cout<<"recognition rate "<<rate<<endl;

	return 0;
}
