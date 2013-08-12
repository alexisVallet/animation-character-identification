#include "DatasetIO.h"

vector<pair<string,Vector2d>, aligned_allocator<pair<string,Vector2d> > > loadFacePositions(string filename) {
	ifstream file(filename);
	vector<pair<string,Vector2d>, aligned_allocator<pair<string,Vector2d> > > facePositions;

	for (CSVIterator line(file); line != CSVIterator(); ++line) {
		string imageName = (*line)[0];
		Vector2d position((double)atoi((*line)[1].c_str()), (double)atoi((*line)[2].c_str()));

		facePositions.push_back(pair<string,Vector2d>(imageName, position));
	}

	return facePositions;
}

void loadDataSet(char* folderName, char** charaNames, int nbImagesPerChara, vector<std::tuple<Mat_<Vec<uchar,3> >,Mat_<float> > > &images, Mat_<int> &classes, vector<DisjointSetForest> &manualSegmentations, vector<pair<int,int> > &facePositions) {
	int nbCharas = 0;
	while (charaNames[nbCharas] != NULL) {
		nbCharas++;
	}
	images = vector<std::tuple<Mat_<Vec<uchar,3> >,Mat_<float> > >(nbCharas * nbImagesPerChara);
	classes = Mat_<int>(nbCharas * nbImagesPerChara, 1);

	// load face positions
	stringstream faceFilename;
	faceFilename<<folderName<<"faceData.csv";
	vector<pair<string,Vector2d>, aligned_allocator<pair<string,Vector2d> > >
		rawFacePositions = loadFacePositions(faceFilename.str());
	facePositions = vector<pair<int,int> >(nbCharas * nbImagesPerChara);

	// load manual segmentations
	manualSegmentations = vector<DisjointSetForest>();
	manualSegmentations.reserve(nbCharas * nbImagesPerChara);

	for (int i = 0; i < nbCharas; i++) {		
		for (int j = 0; j < nbImagesPerChara; j++) {
			stringstream suffix;
			stringstream middle;
			middle<<"_"<<(char)('a' + j);
			suffix<<middle.str()<<".png";
			stringstream fullPath;
			stringstream maskSuffix;
			stringstream segmentationPath;
			maskSuffix<<"-mask.png";
			segmentationPath<<folderName<<charaNames[i]<<middle.str()<<"_seg.png";

			fullPath<<folderName<<charaNames[i]<<suffix.str();
			stringstream maskPath;

			maskPath<<fullPath.str()<<maskSuffix.str();

			int rowMajorIndex = toRowMajor(nbImagesPerChara, j, i);
			Vector2d facePosition = rawFacePositions[rowMajorIndex].second;

			facePositions[rowMajorIndex] = pair<int,int>(
				(int)facePosition(0), (int)facePosition(1));
			Mat_<Vec<uchar, 3> > mask = imread(maskPath.str());
			vector<Mat_<uchar> > maskChannels;

			split(mask, maskChannels);

			get<0>(images[rowMajorIndex]) = imread(fullPath.str());

			Mat_<uchar> thresholdedMask;

			threshold(maskChannels[0], thresholdedMask, 128, 1, THRESH_BINARY_INV);

			get<1>(images[rowMajorIndex]) = Mat_<float>(thresholdedMask);

			//crop(get<0>(images[rowMajorIndex]), get<1>(images[rowMajorIndex]), get<0>(images[rowMajorIndex]), get<1>(images[rowMajorIndex]));

			classes(rowMajorIndex,0) = i;

			// load the manual segmentation
			manualSegmentations.push_back(loadSegmentation(get<1>(images[rowMajorIndex]), segmentationPath.str()));
		}
	}
}
