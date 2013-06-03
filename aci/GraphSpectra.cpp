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

Eigen::SparseMatrix<double> normalizedSparseLaplacian(const WeightedGraph &graph, bool bidirectional, Eigen::VectorXd &degrees) {
	// first we compute the degrees
	degrees = Eigen::VectorXd::Zero(graph.numberOfVertices());

	if (!bidirectional) {
		for (int i = 0; i < graph.getEdges().size(); i++) {
			Edge edge = graph.getEdges()[i];

			degrees(edge.source) += edge.weight;

			if (edge.destination != edge.source) {
				degrees(edge.destination) += edge.weight;
			}
		}
	} else {
		for (int i = 0; i < graph.getEdges().size(); i++) {
			Edge edge = graph.getEdges()[i];

			if (edge.source != edge.destination) {
				degrees(edge.source) += edge.weight;
			} else {
				degrees(edge.source) += edge.weight / 2;
			}
		}
	}

	// then we ccompute triplets for the off diagonal elements while computing the
	// diagonal elements
	vector<Eigen::Triplet<double> > tripletList;
	tripletList.reserve(graph.numberOfVertices() + 2 * graph.getEdges().size());
	Eigen::VectorXd diagonal = Eigen::VectorXd::Ones(graph.numberOfVertices());

	if (!bidirectional) {
		for (int i = 0; i < graph.getEdges().size(); i++) {
			Edge edge = graph.getEdges()[i];
			if (degrees(edge.source) >= 10E-8 && degrees(edge.destination) >= 10E-8) {
				float coeff = -edge.weight/sqrt(degrees(edge.source) * degrees(edge.destination));

				if (edge.source == edge.destination) {
					diagonal(edge.source) += coeff;
				} else {
					tripletList.push_back(Eigen::Triplet<double>(edge.source, edge.destination, coeff));
					tripletList.push_back(Eigen::Triplet<double>(edge.destination, edge.source, coeff));
				}
			}
		}
	} else {
		for (int i = 0; i < graph.getEdges().size(); i++) {
			Edge edge = graph.getEdges()[i];
			if (degrees(edge.source) >= 10E-8 && degrees(edge.destination) >= 10E-8) {
				float coeff = -edge.weight/sqrt(degrees(edge.source) * degrees(edge.destination));

				if (edge.source == edge.destination) {
					diagonal(edge.source) += coeff / 2;
				} else {
					tripletList.push_back(Eigen::Triplet<double>(edge.source, edge.destination, coeff));
				}
			}
		}
	}

	// adding diagonal triplets
	for (int i = 0; i < graph.numberOfVertices(); i++) {
		tripletList.push_back(Eigen::Triplet<double>(i,i,diagonal(i)));
	}

	Eigen::SparseMatrix<double> lapl(graph.numberOfVertices(), graph.numberOfVertices());

	lapl.setFromTriplets(tripletList.begin(), tripletList.end());

	return lapl;
}

Eigen::SparseMatrix<double> randomWalkSparseLaplacian(const WeightedGraph &graph, bool bidirectional, Eigen::VectorXd &degrees) {
	// first we compute the degrees
	degrees = Eigen::VectorXd(graph.numberOfVertices());

	if (!bidirectional) {
		for (int i = 0; i < graph.getEdges().size(); i++) {
			Edge edge = graph.getEdges()[i];

			degrees(edge.source) += edge.weight;
			degrees(edge.destination) += edge.weight;
		}
	} else {
		for (int i = 0; i < graph.getEdges().size(); i++) {
			Edge edge = graph.getEdges()[i];

			degrees(edge.source) += edge.weight;
		}
	}

	// we add the diagonal elements
	vector<Eigen::Triplet<double> > tripletList;
	tripletList.reserve(graph.numberOfVertices() + 2 * graph.getEdges().size());

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		tripletList.push_back(Eigen::Triplet<double>(i,i,1));
	}

	// then we add the off-diagonal elements
	if (!bidirectional) {
		for (int i = 0; i < graph.getEdges().size(); i++) {
			Edge edge = graph.getEdges()[i];
			
			if (degrees(edge.source) >= 10E-8) {
				tripletList.push_back(Eigen::Triplet<double>(edge.source, edge.destination, -edge.weight/degrees(edge.source)));
			}
			if (degrees(edge.destination) >= 10E-8) {
				tripletList.push_back(Eigen::Triplet<double>(edge.destination, edge.source, -edge.weight/degrees(edge.destination)));
			}
		}
	} else {
		for (int i = 0; i < graph.getEdges().size(); i++) {
			Edge edge = graph.getEdges()[i];
			
			if (degrees(edge.source) >= 10E-8) {
				tripletList.push_back(Eigen::Triplet<double>(edge.source, edge.destination, -edge.weight/degrees(edge.source)));
			}
		}
	}

	Eigen::SparseMatrix<double> lapl(graph.numberOfVertices(), graph.numberOfVertices());
	lapl.setFromTriplets(tripletList.begin(), tripletList.end());

	return lapl;
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

// ARPACK routine for non symmetric eigensolver
extern "C" void dnaupd_(int *ido, char *bmat, int *n, char *which,
			int *nev, double *tol, double *resid, int *ncv,
			double *v, int *ldv, int *iparam, int *ipntr,
			double *workd, double *workl, int *lworkl,
			int *info);

// ARPACK routine for sparse eigenvectors computation from dnaupd_'s results
extern "C" void dneupd_(int *rvec, char *howmny, int *select, double *dr,
			double *di, double *z, int *ldz, double *sigmar, double *sigmai,
			double *workev, char *bmat, int *n, char *which,
			int *nev, double *tol, double *resid, int *ncv,
			double *v, int *ldv, int *iparam, int *ipntr,
			double *workd, double *workl, int *lworkl,
			int *info);

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

void nonSymmetricSparseEigenSolver(const Eigen::SparseMatrix<double> &L, char *which, int nev, int maxIterations, Eigen::VectorXd &evalues, Eigen::MatrixXd &evectors) {
	EigenMult mult(&L);

	nonSymmetricSparseEigenSolver(L.rows(), which, nev, maxIterations, evalues, evectors, mult);
}


static inline void generalSparseEigenSolver(bool symmetric, int order, char *which, int nev, int maxIterations, Eigen::VectorXd &evalues, Eigen::MatrixXd &evectors, MatrixVectorMult &mult) {
	//parameters to dnaupd_ / dsaupd_ . See ARPACK's dsaupd (or dnaupd) man page for more info.
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
	int lworkl = symmetric ? (8 + ncv) * ncv : ncv * (6 + 3 * ncv);
	double *workl = new double[lworkl];
	int info = 0;

	// iteratively runnind dsaupd_ / dnaupd_
	while (ido != 99) {
		if (symmetric) {
			dsaupd_(&ido, bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);
		} else {
			dnaupd_(&ido, bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);
		}

		// checking for errors
		if (info != 0 && info != 1) {
			cout<<"dsaupd failed, info = "<<info<<endl;
			cout<<"which = "<<which<<endl;
			cout<<"order = "<<order<<endl;
			cout<<"nev = "<<nev<<endl;
			cout<<"ncv = "<<ncv<<endl;
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

	// running dseupd_ / dneupd_ to figure out eigenvalues and eigenvectors
	// from the results of dsaup
	double *d = new double[symmetric ? 2 * ncv : nev + 1];
	
	if (symmetric) {
		int rvec = 1;
		char howmny[4] = "All";
		int *select = new int[ncv];
		double sigma;
		int ierr;

		dseupd_(&rvec, howmny, select, d, v, &ldv, &sigma, bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &ierr);

		if (ierr != 0) {
			cout<<"dseupd error "<<ierr<<endl;
			exit(EXIT_FAILURE);
		}

		delete[] select;
	} else {
		int rvec = 0;
		char howmny = 'A';
		int *select = new int[ncv];
		double *di = new double[nev + 1];
		double *z = new double[n * (nev + 1)];
		int ldz = max(1, n);
		double sigmar;
		double sigmai;
		double *workev = new double[3 * ncv];

		dneupd_(&rvec, &howmny, select, d, di, z, &ldz, &sigmar, &sigmai, workev,  bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);

		if (info != 0) {
			cout<<"dneupd error "<<info<<endl;
			exit(EXIT_FAILURE);
		}

		delete[] select;
		delete[] di;
		delete[] z;
		delete[] workev;
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
	delete[] d;
}

void symmetricSparseEigenSolver(int order, char *which, int nev, int maxIterations, Eigen::VectorXd &evalues, Eigen::MatrixXd &evectors, MatrixVectorMult &mult) {
	generalSparseEigenSolver(true, order, which, nev, maxIterations, evalues, evectors, mult);
}

void nonSymmetricSparseEigenSolver(int order, char *which, int nev, int maxIterations, Eigen::VectorXd &evalues, Eigen::MatrixXd &evectors, MatrixVectorMult &mult) {
	generalSparseEigenSolver(false, order, which, nev, maxIterations, evalues, evectors, mult);
}