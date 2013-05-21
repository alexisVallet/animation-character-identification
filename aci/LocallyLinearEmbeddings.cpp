#include "LocallyLinearEmbeddings.h"

#define DEBUG_LLE false

/**
 * Matrix multipication routine optimized specifically for the LLE
 * algorithm.
 */
class LLEMult : public MatrixVectorMult {
private:
	const SparseMatrix<double> *W;

public:
	LLEMult(const SparseMatrix<double> *W)
		: W(W)
	{

	}

	void operator() (double *X, double *Y) {
		Map<VectorXd> VX(X, this->W->rows());
		Map<VectorXd> VY(Y, this->W->rows());
		VectorXd XmWX = VX - (*this->W) * VX;
		
		VY = XmWX - this->W->transpose() * XmWX;
	}
};

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

	// determining weights for each sample
	vector<Triplet<double> > tripletList;
	tripletList.reserve(samples.rows * k);

	for (int i = 0; i < samples.rows; i++) {
		Mat_<double> C(k,k);

		for (int u = 0; u < k; u++) {
			for (int v = u; v < k; v++) {
				C(u,v) = (samples.row(i) - samples.row(nearestNeighbors(i,u))).dot(samples.row(i) - samples.row(nearestNeighbors(i,v)));
				C(v,u) = C(u,v);
			}
		}

		// regularize when the number of neighbors is greater than the input
		// dimension
		if (k > samples.cols) {
			C = C + Mat_<double>::eye(k,k) * 10E-3 * trace(C)[0];
		}

		Map<MatrixXd,RowMajor> eigC((double*)C.data, C.rows, C.cols);

		LDLT<MatrixXd> solver(eigC);

		Mat_<double> weights(k,1);
		Map<MatrixXd,RowMajor> eigWeights((double*)weights.data, weights.rows, weights.cols);

		eigWeights = solver.solve(MatrixXd::Ones(k,1));	

		Mat_<double> normWeights;

		double weightsSum = sum(weights)[0];

		if (weightsSum == 0) {
			cout<<"error: cannot reconstruct point "<<i<<" from its neighbors"<<endl;
			exit(EXIT_FAILURE);
		}

		normWeights = weights / weightsSum;

		for (int j = 0; j < k; j++) {
			tripletList.push_back(Triplet<double>(nearestNeighbors(i,j), i, normWeights(j)));
		}
	}

	SparseMatrix<double> W(samples.rows, samples.rows);

	W.setFromTriplets(tripletList.begin(), tripletList.end());

	// constructing vectors in lower dimensional space from the weights
	VectorXd eigenvalues;
	MatrixXd eigenvectors;

	LLEMult mult(&W);

	symmetricSparseEigenSolver(samples.rows, "SM", outDim + 1, samples.rows, eigenvalues, eigenvectors, mult);
	embeddings = Mat_<double>(samples.rows, outDim);

	if (DEBUG_LLE) {
		cout<<"actual eigenvalues : "<<eigenvalues<<endl;
		cout<<"actual : "<<endl<<eigenvectors<<endl;

		MatrixXd denseW(W);
		MatrixXd tmp = MatrixXd::Identity(W.rows(), W.cols()) - denseW;
		MatrixXd M = tmp.transpose() * tmp;
		SelfAdjointEigenSolver<MatrixXd> eigenSolver(M);
		MatrixXd expectedEigenvectors = eigenSolver.eigenvectors();

		cout<<"expected eigenvalues : "<<eigenSolver.eigenvalues()<<endl;
		cout<<"expected : "<<endl<<expectedEigenvectors<<endl;
	}

	for (int i = 0; i < samples.rows; i++) {
		for (int j = 0; j < outDim; j++) {
			embeddings(i,j) = eigenvectors(i, j + 1);
		}
	}
}
