#include "GraphSpectra.h"

Mat_<double> laplacian(const WeightedGraph &graph) {
	Mat_<double> result = Mat_<double>::zeros(graph.numberOfVertices(), graph.numberOfVertices());

	for (int i = 0; i < (int)graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];
		
		result(edge.source, edge.destination) -= edge.weight;
		result(edge.source, edge.source) += edge.weight;

		if (edge.source != edge.destination) {
			result(edge.destination, edge.source) -= edge.weight;
			result(edge.destination, edge.destination) += edge.weight;
		}
	}

	return result;
}

Eigen::SparseMatrix<double> sparseLaplacian(const WeightedGraph &graph, bool bidirectional, Eigen::VectorXd &degrees) {
	// we construct the triplet list without the diagonal first, computing the degrees
	// as we do so, then add the degree triplets.
	degrees = Eigen::VectorXd::Zero(graph.numberOfVertices());
	typedef Eigen::Triplet<double> T;
	vector<T> tripletList;

	tripletList.reserve(graph.getEdges().size());

	if (!bidirectional) {
		for (int i = 0; i < (int)graph.getEdges().size(); i++) {
			Edge edge = graph.getEdges()[i];
			
			if (edge.source != edge.destination) {
				tripletList.push_back(T(edge.source, edge.destination, -edge.weight));
				tripletList.push_back(T(edge.destination, edge.source, -edge.weight));
				degrees(edge.source) += edge.weight;
				degrees(edge.destination) += edge.weight;
			} else {
				degrees(edge.source) -= edge.weight;
			}
		}
	} else {
		for (int i = 0; i < (int)graph.getEdges().size(); i++) {
			Edge edge = graph.getEdges()[i];

			if (edge.source != edge.destination) {
				tripletList.push_back(T(edge.source, edge.destination, -edge.weight));
				degrees(edge.source) += edge.weight;
			} else {
				degrees(edge.source) -= edge.weight / 2;
			}
		}
	}

	// add the diagonal degree elements to the triplet list
	for (int i = 0; i < graph.numberOfVertices(); i++) {
		tripletList.push_back(T(i, i, degrees(i)));
	}

	Eigen::SparseMatrix<double> result(graph.numberOfVertices(), graph.numberOfVertices());

	result.setFromTriplets(tripletList.begin(), tripletList.end());

	return result;
}

Mat_<double> normalizedLaplacian(const WeightedGraph &graph) {
	Mat_<double> unnormalized = laplacian(graph);
	Mat_<double> degrees = Mat_<double>::zeros(graph.numberOfVertices(), graph.numberOfVertices());

	for (int i = 0; i < (int)graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		degrees(edge.source, edge.source) += edge.weight;
		degrees(edge.destination, edge.destination) += edge.weight;
	}

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		degrees(i,i) = 1/sqrt(degrees(i,i));
	}

	return degrees * unnormalized * degrees;
}

Eigen::SparseMatrix<double> normalizedSparseLaplacian(const WeightedGraph &graph, Eigen::VectorXd &degrees) {
	typedef Eigen::Triplet<double> T;
	vector<T> tripletList;

	tripletList.reserve(graph.numberOfVertices() + graph.getEdges().size()/2);

	// first we compute the degrees while initializing the diagonal
	degrees = Eigen::VectorXd::Zero(graph.numberOfVertices());

	for (int i = 0; i < graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		degrees(edge.source) += edge.weight;
	}

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		tripletList.push_back(T(i,i,1));
	}

	// then we compute the rest of the matrix
	for (int i = 0; i < graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		tripletList.push_back(T(edge.source, edge.destination, -edge.weight/sqrt(degrees(edge.source) * degrees(edge.destination))));
	}

	Eigen::SparseMatrix<double> normalized(graph.numberOfVertices(), graph.numberOfVertices());

	normalized.setFromTriplets(tripletList.begin(), tripletList.end());

	return normalized;
}

void packedStorageNormalizedLaplacian(const WeightedGraph &graph, double *L) {
	cout<<"initializing every coefficient to 0"<<endl;
	// we intialize everything to 0
	int n = graph.numberOfVertices();

	for (int i = 0; i < n * (n-1)/2; i++) {
		L[i] = 0;
	}
	cout<<"initializing diagonal"<<endl;
	// we initialize the diagonal to 1
	for (int i = 0; i < graph.numberOfVertices(); i++) {
		int index = toUpperTriangularPacked(i,i);

		if (index < 0 || index >= n * (n + 1) / 2) {
			cout<<"index out of bounds "<<index<<" at ("<<i<<","<<i<<")"<<endl;
		}

		L[index] = 1;
	}

	cout<<"computing degrees"<<endl;
	// we then compute the degrees
	vector<int> degrees(graph.numberOfVertices(), 0);

	for (int i = 0; i < graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		degrees[edge.source] += edge.weight;
		degrees[edge.destination] += edge.weight;
	}

	cout<<"computing non diagonal elements"<<endl;
	// then we compute the non diagonal elements
	for (int i = 0; i < graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		L[toUpperTriangularPacked(edge.source, edge.destination)] = -edge.weight / sqrt((double)degrees[edge.source] * degrees[edge.destination]);
	}
}

// ARPACK routine for lanczos algorithm
extern "C" void dsaupd_(int *ido, char *bmat, int *n, char *which,
			int *nev, double *tol, double *resid, int *ncv,
			double *v, int *ldv, int *iparam, int *ipntr,
			double *workd, double *workl, int *lworkl,
			int *info);

// ARPACK routine for sparse eigenvectors computation from dsaupd_'s results
extern "C" void dseupd_(int *rvec, char *All, int *select, double *d,
			double *v1, int *ldv1, double *sigma, 
			char *bmat, int *n, char *which, int *nev,
			double *tol, double *resid, int *ncv, double *v2,
			int *ldv2, int *iparam, int *ipntr, double *workd,
			double *workl, int *lworkl, int *ierr);

// sparse matrix by vector multiplication: Y = LX where L is an n by n
// sparse matrix, Y and X n sized column vectors.
static void mult(const Eigen::SparseMatrix<double> &L, double *X, double *Y) {
	Eigen::Map<Eigen::VectorXd> VY(Y,L.rows());
	Eigen::Map<Eigen::VectorXd> VX(X,L.rows());

	VY = L * VX;

	for (int i = 0; i < L.rows(); i++) {
		Y[i] = VY(i);
	}
}

void symmetricSparseEigenSolver(const Eigen::SparseMatrix<double> &L, char *which, int nev, Eigen::VectorXd &evalues, Eigen::MatrixXd &evectors) {
	// checking preconditions in debug mode
	assert(L.rows() == L.cols());
	assert(symmetric(L));

	//identity matrix
	typedef Eigen::Triplet<double> T;
	vector<T> triplets;
	triplets.reserve(L.rows());

	for (int i = 0; i < L.rows(); i++) {
		triplets.push_back(T(i,i,1));
	}

	Eigen::SparseMatrix<double> identity(L.rows(), L.cols());

	identity.setFromTriplets(triplets.begin(), triplets.end());

	//parameters to dsaupd_ . See ARPACK's dsaupd man page for more info.
	int ido = 0;
	char bmat[2] = "I";
	int n = L.rows();
	double tol = -1;
	double *resid = new double[n];
	int ncv = min(4*nev,n);
	int ldv = n;
	double *v = new double[ldv * ncv];
	int iparam[11] = {1, 0, 3*n, 1, 2, 0, 1, 0};
	int ipntr[11];
	double *workd = new double[3*n];
	int lworkl = (8 + ncv) * ncv;
	double *workl = new double[lworkl];
	int info = 0;

	// iteratively runnind dsaupd_
	while (ido != 99) {
		dsaupd_(&ido, bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);

		// checking for errors
		if (info != 0) {
			cout<<"dsaupd failed, info = "<<info<<endl;
			exit(EXIT_FAILURE);
		}

		// checking reverse communication flag, performing requested operation
		if (ido == -1) {
			// Y = OP * X
			mult(L, &workd[ipntr[0]-1], &workd[ipntr[1]-1]);
		} else if (ido == 1) {
			// Z = B * X
			mult(identity, &workd[ipntr[0]-1], &workd[ipntr[2]-1]);
			// Y = OP * Z
			mult(L, &workd[ipntr[2]-1], &workd[ipntr[1]-1]);
		} else if (ido == 2) {
			//Y = B * X
			mult(identity, &workd[ipntr[0]-1], &workd[ipntr[1]-1]);
		} else {
			cout<<"unhandled ido = "<<ido<<endl;
		}
	}

	// running dseupd to figure out eigenvalues and eigenvectors
	// from the results of dsaup
	int rvec = 1;
	char howmny[4] = "All";
	int *select = new int[ncv];
	double sigma;
	double *d = new double[2 * ncv];
	int ierr;

	dseupd_(&rvec, howmny, select, d, v, &ldv, &sigma, bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &ierr);

	if (ierr != 0) {
		cout<<"dseupd error "<<ierr<<endl;
		exit(EXIT_FAILURE);
	}

	evalues = Eigen::Map<Eigen::VectorXd>(d,nev);
	evectors = Eigen::Map<Eigen::MatrixXd>(v, L.rows(), nev);

	delete[] resid;
	delete[] v;
	delete[] workd;
	delete[] workl;
	delete[] select;
	delete[] d;
}
