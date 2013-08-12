#include "main.h"

#define TEST true
#define NB_SEGMENTS 30
#define STATFOLDER "../stats/"
#define COLOR_SIGMA 100
#define CENTERS_SIGMA 1
#define NB_EIGENVECTORS 13

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
	vector<std::tuple<Mat_<Vec3b>, Mat_<float> > > dataSet;
	Mat_<int> classes;
	cout<<"loading dataset..."<<endl;
	vector<DisjointSetForest> manualSegmentations;
	loadDataSet("../test/dataset/", charaNames, 5, dataSet, classes, manualSegmentations);

	return 0;
}
