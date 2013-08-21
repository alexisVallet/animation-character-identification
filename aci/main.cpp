#include "main.h"

#define DEBUG true
#define STATFOLDER "../stats/"
#define COLOR_SIGMA 25
#define CENTERS_SIGMA 1
#define AREA_SIGMA 250
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
	char *charaNames[] = {"lupin", NULL};
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
	}

	cout<<"computing matchings..."<<endl;
	MatchingSegmentClassifier classifier(true);
	cout<<"classifier initialized"<<endl;

	for (int i = 0; i < processedDataset.size(); i++) {
		for (int j = i + 1; j < processedDataset.size(); j++) {
			vector<std::tuple<int, int, double> > matching;

			matching = classifier.mostSimilarSegments(
				get<2>(processedDataset[i]), get<0>(processedDataset[i]), get<1>(processedDataset[i]),
				get<2>(processedDataset[j]), get<0>(processedDataset[j]), get<1>(processedDataset[j]));

			vector<Vec3b> colors1(get<2>(processedDataset[i]).getNumberOfComponents());
			vector<Vec3b> colors2(get<2>(processedDataset[j]).getNumberOfComponents());

			for (int k = 0; k < matching.size(); k++) {
				Vec3b randColor((uchar)rand()%255, (uchar)rand()%255, (uchar)rand()%255);
				
				colors1[get<0>(matching[k])] = randColor;
				colors2[get<1>(matching[k])] = randColor;
			}

			imshow("seg1", get<2>(processedDataset[i]).toRegionImage(get<0>(processedDataset[i]), colors1));
			imshow("seg2", get<2>(processedDataset[j]).toRegionImage(get<0>(processedDataset[j]), colors2));
			waitKey(0);
		}
	}

	return 0;
}
