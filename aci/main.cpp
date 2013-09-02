#include "main.h"

#define DEBUG true
#define STATFOLDER "../stats/"
#define COLOR_SIGMA 25
#define CENTERS_SIGMA 1
#define AREA_SIGMA 250
#define NB_EIGENVECTORS 7

void matchingImages(const vector<std::tuple<int,int,double> > &matching, DisjointSetForest &seg1, DisjointSetForest &seg2, const Mat_<Vec3f> &image1, const Mat_<Vec3f> &image2, Mat_<Vec3b> &regionImage1, Mat_<Vec3b> &regionImage2) {
	vector<Vec3b> colors1, colors2;
	colors1.reserve(seg1.getNumberOfComponents());
	colors2.reserve(seg2.getNumberOfComponents());

	for (int i = 0; i < seg1.getNumberOfComponents(); i++) {
		colors1.push_back(Vec3b(0,0,0));
	}

	for (int i = 0; i < seg2.getNumberOfComponents(); i++) {
		colors2.push_back(Vec3b(0,0,0));
	}

	for (int i = 0; i < (int)matching.size(); i++) {
		Vec3b randColor(rand()%255, rand()%255, rand()%255);

		colors1[get<0>(matching[i])] = randColor;
		colors2[get<1>(matching[i])] = randColor;
	}

	regionImage1 = seg1.toRegionImage(image1, colors1);
	regionImage2 = seg2.toRegionImage(image2, colors2);
}

int main(int argc, char** argv) {
	char *charaNames[] = {"rufy", "ray", "miku", "majin", "lupin", "kouji", "jigen", "conan", "chirno", "char", "asuka", "amuro", NULL};
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

	cout<<"embedding"<<endl;
	MatchingSegmentClassifier classifier(true);
	

	typedef std::tuple<DisjointSetForest, Mat_<Vec3f>, Mat_<float> > Sample;

	vector<Sample> samples;
	samples.reserve(dataset.size());

	for (int i = 0; i < (int)dataset.size(); i++) {
		samples.push_back(Sample(segmentations[i], get<0>(processedDataset[i]), get<1>(processedDataset[i])));
	}

	MatrixXd similarity;
	classifier.similarityMatrix(samples, similarity);
	VectorXi classLabels;
	
	spectralClustering(DenseSimilarityMatrix(&similarity), CompleteGraph(), _sparseLaplacian, 12, classLabels);

	cout<<classLabels.transpose()<<endl;

	return 0;
}
