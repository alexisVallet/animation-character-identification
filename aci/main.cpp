#include "main.h"

#define DEBUG true
#define STATFOLDER "../stats/"
#define COLOR_SIGMA 100
#define CENTERS_SIGMA 1
#define AREA_SIGMA 1500
#define NB_EIGENVECTORS 7

DisjointSetForest addBackgroundSegment(DisjointSetForest segmentation, const Mat_<float> &mask) {
	// compute indexes from the image mask
	vector<int> indexes(mask.rows * mask.cols, -1);
	int k = 0;

	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (mask(i,j) > 0) {
				indexes[toRowMajor(mask.cols, j, i)] = k;
				k++;
			}
		}
	}

	DisjointSetForest newSegmentation(indexes.size());
	int backgroundRep = -1;
	vector<int> segmentReps(segmentation.getNumberOfComponents());
	map<int,int> rootIndexes = segmentation.getRootIndexes();
	
	for (int i = 0; i < (int)indexes.size(); i++) {
		// if the pixel is part of the background, fuse with backgroundRep
		if (indexes[i] < 0) {
			if (backgroundRep < 0) {
				backgroundRep = i;
			} else {
				newSegmentation.setUnion(i, backgroundRep);
			}
		} else { // otherwise, fuse with the corresponding class rep
			int rep = segmentReps[rootIndexes[segmentation.find(indexes[i])]];

			if (rep < 0) {
				segmentReps[rootIndexes[segmentation.find(indexes[i])]] = i;
			} else {
				newSegmentation.setUnion(rep, i);
			}
		}
	}

	return newSegmentation;
}

int main(int argc, char** argv) {
	char *charaNames[] = {"lupin", "rufy", NULL};
	vector<std::tuple<Mat_<Vec3b>, Mat_<float> > > dataset;
	Mat_<int> classes;
	cout<<"loading dataset..."<<endl;
	vector<Mat_<Vec3b> > manualSegmentations;
	loadDataSet("../test/dataset/", charaNames, 5, dataset, classes, manualSegmentations);

	cout<<"preprocessing and loading manual segmentations..."<<endl;
	vector<std::tuple<Mat_<Vec3b>, Mat_<float>, DisjointSetForest > > processedDataset;
	processedDataset.reserve(dataset.size());

	for (int i = 0; i < dataset.size(); i++) {
		std::tuple<Mat_<Vec3b>, Mat_<float>, DisjointSetForest> processedSample;
		Mat_<Vec3b> segmentationImage;

		preProcessing(get<0>(dataset[i]), get<1>(dataset[i]), get<0>(processedSample), get<1>(processedSample), manualSegmentations[i], segmentationImage);

		get<2>(processedSample) = segmentationImageToSegmentation(segmentationImage, get<1>(processedSample));

		processedDataset.push_back(processedSample);

		if (DEBUG) {
			vector<VectorXd> averageColor = averageColorLabeling(get<2>(processedSample), get<0>(processedSample), get<1>(processedSample));
			vector<Vec3b> cvColors;
			cvColors.reserve(averageColor.size());

			for (int j = 0; j < averageColor.size(); j++) {
				Vec3b cvColor((uchar)averageColor[j](0), (uchar)averageColor[j](1), (uchar)averageColor[j](2));

				cout<<cvColor<<", ";

				cvColors.push_back(cvColor);
			}
			cout<<endl;

			Mat_<Vec3b> regionImage = get<2>(processedSample).toRegionImage(get<0>(processedSample), cvColors);

			imshow("segmentation", regionImage);
			waitKey(0);
		}
	}

	cout<<"classification"<<endl;
	float rate = 0;

	for (int i = 0; i < dataset.size(); i++) {
		cout<<"classifying sample "<<i<<endl;
		vector<std::tuple<SegmentLabeling, double> > features;

		features.push_back(std::tuple<SegmentLabeling, double>(&averageColorLabeling, COLOR_SIGMA));
		features.push_back(std::tuple<SegmentLabeling, double>(&gravityCenterLabeling, CENTERS_SIGMA));
		features.push_back(std::tuple<SegmentLabeling, double>(&segmentAreaLabeling, AREA_SIGMA));

		MultipleGraphsClassifier classifier(features, NB_EIGENVECTORS);

		cout<<"building training set"<<endl;
		vector<std::tuple<DisjointSetForest, Mat_<Vec3b>, Mat_<float>, int > > trainingSet;
		trainingSet.reserve(dataset.size() - 1);

		for (int j = 0; j < dataset.size(); j++) {
			if (i != j) {
				std::tuple<DisjointSetForest, Mat_<Vec3b>, Mat_<float>, int> trainingSample;
				get<0>(trainingSample) = get<2>(processedDataset[j]);
				get<1>(trainingSample) = get<0>(processedDataset[j]);
				get<2>(trainingSample) = get<1>(processedDataset[j]);
				get<3>(trainingSample) = classes(j, 0);

				trainingSet.push_back(trainingSample);
			}
		}

		cout<<"training"<<endl;
		classifier.train(trainingSet);
		cout<<"predicting"<<endl;
		int predictedClass = classifier.predict(get<2>(processedDataset[i]), get<0>(processedDataset[i]), get<1>(processedDataset[i]));

		if (predictedClass == classes(i,0)) {
			rate += 1./(double)dataset.size();
		}

		cout<<"actual: "<<classes(i,0)<<", predicted: "<<predictedClass<<endl;
	}

	cout<<"recognition rate: "<<rate<<endl;

	return 0;
}
