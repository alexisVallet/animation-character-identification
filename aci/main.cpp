#include "main.h"

#define TEST true
#define NB_SEGMENTS 30
#define STATFOLDER "../stats/"

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
	if (TEST) {
		testPatternVectors();
	} else {
		cout<<"loading dataset..."<<endl;
		char *charaNames[] = {"rufy", "ray", "miku", "majin", "lupin", "kouji", "jigen", "conan", "chirno", "char", "asuka", "amuro", NULL};
		vector<std::tuple<Mat_<Vec3b>, Mat_<float>, pair<int,int> > > dataSet;
		Mat_<int> classes;

		loadDataSet("../test/dataset/", charaNames, 5, dataSet, classes);

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
		vector<std::tuple<DisjointSetForest, WeightedGraph> > segmentations;
		segmentations.reserve(dataSet.size());

		for (int i = 0; i < (int)dataSet.size(); i++) {
			std::tuple<DisjointSetForest, WeightedGraph> segmentation;

			segment(processedDataSet[i].first, processedDataSet[i].second, get<0>(segmentation), get<1>(segmentation));

			segmentations.push_back(segmentation);
		}
	}

	return 0;
}
