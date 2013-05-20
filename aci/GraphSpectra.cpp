#include "GraphSpectra.h"

Mat_<double> laplacian(const WeightedGraph &graph) {
	Mat_<double> result = Mat_<double>::zeros(graph.numberOfVertices(), graph.numberOfVertices());

	for (int i = 0; i < graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		result(edge.source, edge.destination) = -edge.weight;
		result(edge.destination, edge.source) = -edge.weight;
		result(edge.source, edge.source) += edge.weight;
		result(edge.destination, edge.destination) += edge.weight;
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
	assert(noLoops(graph));
	assert(bidirectional(graph));
	// We first compute the degree of each vertex while initializing the diagonal
	// triplets.
	degrees = Eigen::VectorXd::Zero(graph.numberOfVertices());
	typedef Eigen::Triplet<double> T;
	vector<T> triplets;
	// the diagonal is non zero + one non zero element per edge, divided by 2 because of bidirectional rep
	triplets.reserve(graph.numberOfVertices() + graph.getEdges().size()/2);

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		double selfLoopWeight = 0;

		for (int j = 0; j < (int)graph.getAdjacencyList(i).size(); j++) {
			HalfEdge edge = graph.getAdjacencyList(i)[j];

			degrees(i) += edge.weight;

			if (i == edge.destination) {
				selfLoopWeight += edge.weight;
			}
		}

		triplets.push_back(T(i,i,1 - (degrees(i) != 0 ? selfLoopWeight / degrees(i) : 0)));
	}

	// Then we compute the coefficient for each edge
	for (int i = 0; i < (int)graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];
		double denominator = sqrt(degrees(edge.source) * degrees(edge.destination));

		// only adds a triplet if the weights and degrees are non zero
		// to avoid NaN due to 0/0. This may happen in weighted graphs with
		// edges weighted to 0 or very close to 0.
		if (edge.weight > 0 && denominator > 0) {
			// only add coeff if it is absolutely greater than an arbitrary epsilon
			double coeff = -edge.weight / denominator;

			if (abs(coeff) > 10E-8) {
				triplets.push_back(T(edge.source, edge.destination, coeff));
			}
		}
	}

	Eigen::SparseMatrix<double> normalized(graph.numberOfVertices(), graph.numberOfVertices());

	normalized.setFromTriplets(triplets.begin(), triplets.end());

	return normalized;
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

EigenMult::EigenMult(const Eigen::SparseMatrix<double> *L) 
	: L(L)
{

}

// sparse matrix by vector multiplication: Y = LX where L is an n by n
// sparse matrix, Y and X n sized column vectors.
void EigenMult::operator() (double *X, double *Y) {
	Eigen::Map<Eigen::VectorXd> VX(X, this->L->rows());
	Eigen::Map<Eigen::VectorXd> VY(Y, this->L->rows());

	VY = (*L) * VX;
}

void symmetricSparseEigenSolver(const Eigen::SparseMatrix<double> &L, char *which, int nev, int maxIterations, Eigen::VectorXd &evalues, Eigen::MatrixXd &evectors) {
	EigenMult mult(&L);

	symmetricSparseEigenSolver(L.rows(), which, nev, maxIterations, evalues, evectors, mult);
}

void symmetricSparseEigenSolver(int order, char *which, int nev, int maxIterations, Eigen::VectorXd &evalues, Eigen::MatrixXd &evectors, MatrixVectorMult &mult) {
	//parameters to dsaupd_ . See ARPACK's dsaupd man page for more info.
	int ido = 0;
	char bmat[2] = "I";
	int n = order;
	double tol = -1;
	double *resid = new double[n];
	int ncv = min(4*nev,n);
	int ldv = n;
	double *v = new double[ldv * ncv];
	int iparam[11] = {1, 0, maxIterations, 1, 2, 0, 1, 0};
	int ipntr[11];
	double *workd = new double[3*n];
	int lworkl = (8 + ncv) * ncv;
	double *workl = new double[lworkl];
	int info = 0;

	// iteratively runnind dsaupd_
	while (ido != 99) {
		dsaupd_(&ido, bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);

		// checking for errors
		if (info != 0 && info != 1) {
			cout<<"dsaupd failed, info = "<<info<<endl;
			exit(EXIT_FAILURE);
		}

		// checking reverse communication flag, performing requested operation
		if (ido == -1) {
			// Y = OP * X
			mult(&workd[ipntr[0]-1], &workd[ipntr[1]-1]);
		} else if (ido == 1) {
			// Y = OP * Z
			mult(&workd[ipntr[2]-1], &workd[ipntr[1]-1]);
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
	// copying values to output arrays
	evalues = Eigen::VectorXd(nev);
	evectors = Eigen::MatrixXd(order, nev);

	for (int i = 0; i < nev; i++) {
		evalues(i) = d[i];

		for (int j = 0; j < order; j++) {
			evectors(j,i) = v[toColumnMajor(order, j, i)];
		}
	}

	delete[] resid;
	delete[] v;
	delete[] workd;
	delete[] workl;
	delete[] select;
	delete[] d;
}
