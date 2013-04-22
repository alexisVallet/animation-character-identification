#include "NormalizedCuts.h"

/**
 * Uses the lanczos algorithm to compute a tridiagonal matrix similar
 * to the argument.
 *
 * @param M square matrix to compute tridiagonal matrix from
 * @param alpha vector of diagonal values for the tridiagonal matrix
 * @param beta vector of out of diagonal values for the tridiagonal matrix
 * @param Vm output matrix used for eigenvector determination
 */
void lanczosAlgorithm(SparseMatrix<double> &M, VectorXd &alpha, VectorXd &beta) {
	alpha = VectorXd(M.rows());
	beta = VectorXd(M.rows() - 1);
	VectorXd beta_(M.rows() + 1);
	beta_(0) = 0; // trick to avoid branching
	VectorXd v[] = {VectorXd(M.rows()), VectorXd(M.rows())};
	
	cout<<"initializing random initial vector"<<endl;
	for (int i = 0; i < M.rows(); i++) {
		v[0](i) = (double)rand() / RAND_MAX;
	}

	cout<<"normalizing initial vector"<<endl;
	v[0].normalize();

	VectorXd w(M.rows());

	// v[j%2] is the current vector, v[(j+1)%2] is the one from the previous iteration,
	// or the next.
	for (int j = 0; j < M.rows(); j++) {
		int current = j%2;
		int other = 1 - current;

		w = M * v[current];

		alpha(j) = w.dot(v[current]);
		w = w - alpha(j) * v[current] - beta_(j) * v[other];

		for (int k = 0; k < w.size(); k++) {
			double wk = w(k);

			beta_(j + 1) += wk * wk;
		}

		v[other] = w / beta_(j + 1);
	}

	cout<<"copying beta"<<endl;
	beta = beta_.tail(M.rows()).head(M.rows() - 1);
}

/**
 * Computes the QR decomposition of a tridiagonal matrix. Algorithm by James Ortega and Henry Kaiser, 1963.
 *
 * @param alpha1 input vector of the diagonal elements of the input matrix
 * @param beta1 input vector of the squared off-diagonal elements of the input matrix
 * @param alpha2 output vector of the diagonal of the A' matrix such that
 * A' = R * Q where A = Q * R, A being the input matrix.
 * @param beta2 output vector of the off-diagonal elements of A'
 */
void tridiagonalQRDecomposition(const VectorXd &alpha1, const VectorXd &beta1, VectorXd &alpha2, VectorXd &beta2) {
	assert(alpha1.size() == beta1.size() + 1);
	cout<<"initializing vectors a and b"<<endl;
	VectorXd a(alpha1.size() + 2);
	a.head(alpha1.size()) = alpha1;
	a(alpha1.size() + 1) = 0;
	VectorXd b(beta1.size() + 1);
	b.head(beta1.size()) = beta1;
	double u = 0;
	double s = 0;

	alpha2 = VectorXd(alpha1.size());
	beta2 = VectorXd(beta1.size());

	for (int i = 1; i <= alpha1.size(); i++) {
		cout<<"iteration "<<i<<endl;
		double gamma = a(i) - u - 1;
		double p;

		if (s != 1) {
			p = (gamma * gamma) / (1 - s);
		} else {
			p = (1 - s) * b(i-1);
		}

		if (i != 1) {
			beta2(i - 2) = s * (p + b(i));
		}

		s = b(i) / (p + b(i));
		u = s * (gamma + a(i + 1));
		alpha2(i - 1) = gamma + u;
	}
}

/**
 * Uses the QR algorithm to determine eigenvalues of a tridiagonal matrix.
 *
 * @param alpha input vector of diagonal values of the tridiagonal matrix.
 * @param beta input vector of the squared off-diagonal values of the tridiagonal matrix.
 * @param eigenvalues output vector of eigenvalues of the matrix.
 */
void qrAlgorithm(const VectorXd &alpha, const VectorXd &beta, VectorXd &eigenvalues) {
	assert(alpha.size() == beta.size() + 1);

	// then we decompose until convergence
	VectorXd beta1 = beta;
	VectorXd alpha1 = alpha;
	VectorXd alpha2(alpha.size());
	VectorXd beta2(beta.size());

	do {
		cout<<"computing tridiagonal QR decomposition"<<endl;
		tridiagonalQRDecomposition(alpha1, beta1, alpha2, beta2);
		cout<<"computed!"<<endl;

		if (!(alpha1.isApprox(alpha2) && beta1.isApprox(beta2))) {
			alpha1 = alpha2;
			beta1 = beta2;
		} else  {
			break;
		}
	} while (1);

	eigenvalues = alpha1;
}

static void swap(VectorXd &list, int i, int j) {
	double tmp = list(i);
	list(i) = list(j);
	list(j) = tmp;
}

int selectPartition(VectorXd &list, int left, int right, int pivot) {
	double pivotValue = list(pivot);
	swap(list, pivot, right);
	int storeIndex = left;

	for (int i = left; i < right; i++) {
		if (list(i) < pivotValue) {
			swap(list, storeIndex, i);
			storeIndex++;
		}
	}

	swap(list, right, storeIndex);

	return storeIndex;
}

/**
 * Finds the k-th smallest element within a list using Hoare's algorithm.
 */
int select(VectorXd &list, int left, int right, int k) {
	if (left == right) {
		return left;
	}
	int pivot = (left + right) / 2;
	int newPivot = selectPartition(list, left, right, pivot);
	int pivotDist = newPivot - left + 1;

	if (pivotDist == k) {
		return newPivot;
	} else if (k < pivotDist) {
		return select(list, left, newPivot - 1, k);
	} else {
		return select(list, newPivot + 1, right, k - pivotDist);
	}
}

/**
 * Returns the eigenvector corresponding to the second smallest eigenvalue of
 * a square symmetric real matrix L.
 *
 * @param L matrix to compute the eigenvector of.
 * @return the eigenvector corresponding to the second smallest eigenvalue of L.
 */
VectorXd secondSmallestEigenvector(SparseMatrix<double> L) {
	VectorXd alpha1, beta1, eigenvalues;

	cout<<"computing tridiagonal matrix"<<endl;
	lanczosAlgorithm(L, alpha1, beta1);

	cout<<"computing eigenvalues from tridiagonal matrix"<<endl;
	qrAlgorithm(alpha1, beta1, eigenvalues);

	cout<<"extracting second smallest eigenvalue"<<endl;
	int secondSmallestIndex = select(eigenvalues, 0, eigenvalues.size() - 1, 2);

	cout<<"building matrix for linear system solving"<<endl;
	SparseMatrix<double> diagEv2(L.rows(), L.cols());
	vector<Triplet<double> > tripletList;
	tripletList.reserve(L.rows());

	for (int i = 0; i < L.rows(); i++) {
		tripletList.push_back(Triplet<double>(i,i,eigenvalues(secondSmallestIndex)));
	}

	diagEv2.setFromTriplets(tripletList.begin(), tripletList.end());

	SparseMatrix<double> L_ = L - diagEv2;

	cout<<"initializing linear system"<<endl;
	SparseLU<SparseMatrix<double>, COLAMDOrdering<int> > linSolver(L_);

	cout<<"solving linear system"<<endl;
	return linSolver.solve(VectorXd::Zero(L.rows()));
}

DisjointSetForest normalizedCuts(WeightedGraph &graph, double stop) {
	cout<<"computing normalized laplacian"<<endl;
	SparseMatrix<double> L = normalizedSparseLaplacian(graph);

	cout<<"computing eigenvector with second smallest eigenvalue"<<endl;
	VectorXd x = secondSmallestEigenvector(L);

	cout<<x<<endl;

	return DisjointSetForest(0);
}