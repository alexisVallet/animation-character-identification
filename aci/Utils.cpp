#include "Utils.hpp"

int toRowMajor(int width, int x, int y) {
  return x + width * y;
}

pair<int,int> fromRowMajor(int width, int i) {
  pair<int,int> coords(i/width, i%width);

  return coords;
}

void loadDataSet(char* folderName, char** charaNames, int nbCharas, int nbImagesPerChara, vector<pair<Mat_<Vec<uchar,3> >,Mat_<float> > > &images, Mat &classes) {
	images = vector<pair<Mat_<Vec<uchar,3> >,Mat_<float> > >(nbCharas * nbImagesPerChara);
	classes = Mat(nbCharas * nbImagesPerChara, 1, CV_32S);

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

			Mat_<Vec<uchar, 3> > mask = imread(maskPath);
			vector<Mat_<uchar> > maskChannels;

			split(mask, maskChannels);

			images[rowMajorIndex].first = imread(fullPath);
			images[rowMajorIndex].second = Mat_<float>::ones(mask.rows, mask.cols) - (Mat_<float>(maskChannels[0]) / 255);
			waitKey(0);
			classes.at<int>(rowMajorIndex,0) = i;

			delete[] fullPath;
			delete[] maskPath;
		}
	}
}

Mat_<double> sparseMul(SparseMat_<double> A, Mat_<double> b) {
	assert(A.size(1) == b.rows);
	assert(b.cols == 1);
	Mat_<double> c = Mat_<double>::zeros(b.rows, 1);

	SparseMatConstIterator_<double> it;

	// iterates over non zero elements
	for (it = A.begin(); it != A.end(); ++it) {
		const SparseMat_<double>::Node* n = it.node();
		int row = n->idx[0];
		int col = n->idx[1];

		c(row, 0) += it.value<double>() * b(col,0);
	}

	return c;
}

bool symmetric(Eigen::SparseMatrix<double> M) {
	bool res = true;

	for (int k = 0; k < M.outerSize(); k++) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(M,k); it; ++it) {
			res = res && abs(it.value() - M.coeffRef(it.col(), it.row())) <= 10E-8;
		}
	}

	return res;
}

bool positiveDefinite(Eigen::SparseMatrix<double> M) {
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > chol;

	chol.compute(M);

	return chol.info() == Eigen::Success;
}

// Remove a line and column with the same index in a sparse matrix.
void removeLineCol(const Eigen::SparseMatrix<double> &L, int v0, Eigen::SparseMatrix<double> &L0) {
	typedef Eigen::Triplet<double> T;
	vector<T> tripletList;
	tripletList.reserve(L.nonZeros());

	for (int k = 0; k < L.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(L,k); it; ++it) {
			if (it.row() != v0 && it.col() != v0) {
				int newRow = it.row() < v0 ? it.row() : it.row() - 1;
				int newCol = it.col() < v0 ? it.col() : it.col() - 1;

				tripletList.push_back(T(newRow, newCol, it.value()));
			}
		}
	}

	L0 = Eigen::SparseMatrix<double>(L.rows() - 1, L.cols() - 1);

	L0.setFromTriplets(tripletList.begin(), tripletList.end());
}

int toUpperTriangularPacked(int i, int j) {
	if (i > j) {
		return toUpperTriangularPacked(j, i);
	} else {
		int result = i + (j + 1) * j / 2;

		return result;
	}
}