#include "LocallyLinearEmbeddings.h"

void locallyLinearEmbeddings(const Mat_<float> &samples, int outDim, Mat_<float> &embeddings, int k) {
	assert(outDim < samples.cols);
	assert(k >= 1);
	Mat_<int> nearestNeighbors(samples.rows, k);

	// determining k nearest neighbors for each sample
	flann::Index flannIndex(samples, flann::LinearIndexParams());

	for (int i = 0; i < samples.rows; i++) {
		Mat_<int> nearest;
		Mat dists;

		flannIndex.knnSearch(samples.row(i), nearest, dists, k + 1);
		nearest.colRange(1, nearest.cols).copyTo(nearestNeighbors.row(i));
	}

	cout<<"nn : "<<nearestNeighbors<<endl;

	// determining weights for each sample
	vector<Triplet<double> > tripletList;
	tripletList.reserve(samples.rows * k);

	for (int i = 0; i < samples.rows; i++) {
		Mat_<double> X = Mat_<double>::zeros(samples.cols, k);

		for (int j = 0; j < k; j++) {
			X.col(j) = (samples.row(i) - samples.row(nearestNeighbors(i,j))).t();
		}

		Mat_<double> A = X.t() * X;
		Mat_<double> Ap;

		Mat_<double> weights;
	
		Map<MatrixXd,RowMajor> eigA((double*)A.data, A.rows, A.cols);

		ColPivHouseholderQR<MatrixXd> solver(eigA, MatrixXd::Ones(k));

		for (int j = 0; j < k; j++) {
			tripletList.push_back(Triplet<double>(i, nearestNeighbors(i,j), weights(j)));
		}
	}

	SparseMatrix<double> W(samples.rows, samples.rows);

	W.setFromTriplets(tripletList.begin(), tripletList.end());

	cout<<"W: "<<endl<<W<<endl;
}
