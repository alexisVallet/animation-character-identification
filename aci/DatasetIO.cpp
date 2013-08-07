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

void loadDataSet(char* folderName, char** charaNames, int nbImagesPerChara, vector<tuple<Mat_<Vec<uchar,3> >,Mat_<float>, pair<int,int> > > &images, Mat_<int> &classes) {
	int nbCharas = 0;
	while (charaNames[nbCharas] != NULL) {
		nbCharas++;
	}
	images = vector<tuple<Mat_<Vec<uchar,3> >,Mat_<float>, pair<int,int> > >(nbCharas * nbImagesPerChara);
	classes = Mat_<int>(nbCharas * nbImagesPerChara, 1);

	// load face positions
	stringstream faceFilename;
	faceFilename<<folderName<<"faceData.csv";
	vector<pair<string,Vector2d>, aligned_allocator<pair<string,Vector2d> > >
		facePositions = loadFacePositions(faceFilename.str());

	for (int i = 0; i < nbCharas; i++) {		
		for (int j = 0; j < nbImagesPerChara; j++) {
			char suffix[] = {'_', 'a' + j, '.', 'p', 'n', 'g', '\0'};
			char *fullPath = new char[strlen(folderName) + strlen(charaNames[i]) + strlen(suffix) + 1];
			char maskSuffix[] = {'-', 'm', 'a', 's', 'k', '.', 'p', 'n', 'g', '\0'};
			char *maskPath = new char[strlen(folderName) + strlen(charaNames[i]) + strlen(suffix) + strlen(maskSuffix) + 1];
			
			strcpy(fullPath, folderName);
			strcat(fullPath, charaNames[i]);
			strcat(fullPath, suffix);
			strcpy(maskPath, fullPath);
			strcat(maskPath, maskSuffix);

			int rowMajorIndex = toRowMajor(nbImagesPerChara, j, i);

			Vector2d facePosition = facePositions[rowMajorIndex].second;

			get<2>(images[rowMajorIndex]) = pair<int,int>(
				(int)facePosition(0), (int)facePosition(1));

			Mat_<Vec<uchar, 3> > mask = imread(maskPath);
			vector<Mat_<uchar> > maskChannels;

			split(mask, maskChannels);

			get<0>(images[rowMajorIndex]) = imread(fullPath);

			Mat_<uchar> thresholdedMask;

			threshold(maskChannels[0], thresholdedMask, 128, 1, THRESH_BINARY_INV);

			get<1>(images[rowMajorIndex]) = Mat_<float>(thresholdedMask);

			//crop(get<0>(images[rowMajorIndex]), get<1>(images[rowMajorIndex]), get<0>(images[rowMajorIndex]), get<1>(images[rowMajorIndex]));

			classes(rowMajorIndex,0) = i;

			delete[] fullPath;
			delete[] maskPath;
		}
	}
}
