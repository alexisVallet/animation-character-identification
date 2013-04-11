#include "Utils.hpp"

int toRowMajor(int width, int x, int y) {
  return x + width * y;
}

pair<int,int> fromRowMajor(int width, int i) {
  pair<int,int> coords(i/width, i%width);

  return coords;
}

void loadDataSet(char* folderName, char** charaNames, int nbCharas, int nbImagesPerChara, vector<Mat> &images, Mat &classes) {
	images = vector<Mat>(nbCharas * nbImagesPerChara);
	classes = Mat(nbCharas * nbImagesPerChara, 1, CV_32S);

	for (int i = 0; i < nbCharas; i++) {		
		for (int j = 0; j < nbImagesPerChara; j++) {
			char* fullPath = new char[strlen(folderName) + strlen(charaNames[i]) + 7];
			char suffix[7] = {'_', 'a' + j, '.', 'p', 'n', 'g', '\0'};
			
			strcpy(fullPath, folderName);
			strcat(fullPath, charaNames[i]);
			strcat(fullPath, suffix);

			int rowMajorIndex = toRowMajor(nbImagesPerChara, j, i);

			images[rowMajorIndex] = imread(fullPath, -1);
			classes.at<int>(rowMajorIndex,0) = i;

			delete[] fullPath;
		}
	}
}

Mat_<double> sparseMul(SparseMat_<double> A, Mat_<double> b) {
	assert(A.cols == b.rows);
	assert(b.cols == 1);
	Mat_<double> c = Mat_<double>::zeros(b.rows, 1);

	SparseMatConstIterator_<double> it;

	// iterates over non zero elements
	for (it = A.begin(); it != A.end(); it++) {
		const SparseMat_<double>::Node* n = it.node();
		int row = n->idx[0];
		int col = n->idx[1];

		c(col, 0) += it.value<double>() * b(col,0);
	}

	return c;
}