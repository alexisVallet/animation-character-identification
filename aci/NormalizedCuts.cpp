#include "NormalizedCuts.h"

static SparseMatrix<double> bigMatrix;

void av(int n, double *in, double *out) {
	cout<<"calling av"<<endl;
	VectorXd inVec = Map<VectorXd>(in, n, 1);
	VectorXd outVec = bigMatrix * inVec;
	double *outData = outVec.data();

	memcpy(out, outData, n * sizeof(double));
}

extern "C" void dsaupd_(int *ido, char *bmat, int *n, char *which,
			int *nev, double *tol, double *resid, int *ncv,
			double *v, int *ldv, int *iparam, int *ipntr,
			double *workd, double *workl, int *lworkl,
			int *info);

extern "C" void dseupd_(int *rvec, char *All, int *select, double *d,
			double *v1, int *ldv1, double *sigma, 
			char *bmat, int *n, char *which, int *nev,
			double *tol, double *resid, int *ncv, double *v2,
			int *ldv2, int *iparam, int *ipntr, double *workd,
			double *workl, int *lworkl, int *ierr);

void dsaupd(int n, int nev, double *Evals)
{
  int ido = 0; /* Initialization of the reverse communication
		  parameter. */

  char bmat[2] = "I"; /* Specifies that the right hand side matrix
			 should be the identity matrix; this makes
			 the problem a standard eigenvalue problem.
			 Setting bmat = "G" would have us solve the
			 problem Av = lBv (this would involve using
			 some other programs from BLAS, however). */

  char which[3] = "SM"; /* Ask for the nev eigenvalues of smallest
			   magnitude.  The possible options are
			   LM: largest magnitude
			   SM: smallest magnitude
			   LA: largest real component
			   SA: smallest real compoent
			   LI: largest imaginary component
			   SI: smallest imaginary component */

  double tol = 0.0; /* Sets the tolerance; tol<=0 specifies 
		       machine precision */

  double *resid;
  resid = new double[n];

  int ncv = 4*nev; /* The largest number of basis vectors that will
		      be used in the Implicitly Restarted Arnoldi
		      Process.  Work per major iteration is
		      proportional to N*NCV*NCV. */
  if (ncv>n) ncv = n;

  double *v;
  int ldv = n;
  v = new double[ldv*ncv];

  int *iparam;
  iparam = new int[11]; /* An array used to pass information to the routines
			   about their functional modes. */
  iparam[0] = 1;   // Specifies the shift strategy (1->exact)
  iparam[2] = 3*n; // Maximum number of iterations
  iparam[6] = 1;   /* Sets the mode of dsaupd.
		      1 is exact shifting,
		      2 is user-supplied shifts,
		      3 is shift-invert mode,
		      4 is buckling mode,
		      5 is Cayley mode. */

  int *ipntr;
  ipntr = new int[11]; /* Indicates the locations in the work array workd
			  where the input and output vectors in the
			  callback routine are located. */

  double *workd;
  workd = new double[3*n];

  double *workl;
  workl = new double[ncv*(ncv+8)];

  int lworkl = ncv*(ncv+8); /* Length of the workl array */

  int info = 0; /* Passes convergence information out of the iteration
		   routine. */

  int rvec = 0; /* Specifies that eigenvectors should not be calculated */

  int *select;
  select = new int[ncv];
  double *d;
  d = new double[2*ncv]; /* This vector will return the eigenvalues from
			    the second routine, dseupd. */
  double sigma;
  int ierr;

  /* Here we enter the main loop where the calculations are
     performed.  The communication parameter ido tells us when
     the desired tolerance is reached, and at that point we exit
     and extract the solutions. */

  do {
    dsaupd_(&ido, bmat, &n, which, &nev, &tol, resid, 
	    &ncv, v, &ldv, iparam, ipntr, workd, workl,
	    &lworkl, &info);
    
    if ((ido==1)||(ido==-1)) av(n, workd+ipntr[0]-1, workd+ipntr[1]-1);
  } while ((ido==1)||(ido==-1));

  /* From those results, the eigenvalues and vectors are
     extracted. */

  if (info<0) {
         cout << "Error with dsaupd, info = " << info << "\n";
         cout << "Check documentation in dsaupd\n\n";
  } else {
    dseupd_(&rvec, "All", select, d, v, &ldv, &sigma, bmat,
	    &n, which, &nev, &tol, resid, &ncv, v, &ldv,
	    iparam, ipntr, workd, workl, &lworkl, &ierr);

    if (ierr!=0) {
      cout << "Error with dseupd, info = " << ierr << "\n";
      cout << "Check the documentation of dseupd.\n\n";
    } else if (info==1) {
      cout << "Maximum number of iterations reached.\n\n";
    } else if (info==3) {
      cout << "No shifts could be applied during implicit\n";
      cout << "Arnoldi update, try increasing NCV.\n\n";
    }
    
    /* Before exiting, we copy the solution information over to
       the arrays of the calling program, then clean up the
       memory used by this routine.  For some reason, when I
       don't find the eigenvectors I need to reverse the order of
       the values. */

    int i;
    for (i=0; i<nev; i++) Evals[i] = d[nev-1-i];

    delete resid;
    delete v;
    delete iparam;
    delete ipntr;
    delete workd;
    delete workl;
    delete select;
    delete d;
  }
}


void dsaupd(int n, int nev, double *Evals, double **Evecs)
{
  cout<<"ido"<<endl;
  int ido = 0;
  cout<<"bmat"<<endl;
  char bmat[2] = "I";
  cout<<"which"<<endl;
  char which[3] = "SM";
  cout<<"tol"<<endl;
  double tol = 0.0;
  cout<<"resid"<<endl;
  double *resid;
  resid = new double[n];
  cout<<"ncv"<<endl;
  int ncv = 4*nev;
  if (ncv>n) ncv = n;
  cout<<"v"<<endl;
  double *v;
  int ldv = n;
  v = new double[ldv*ncv];
  cout<<"iparam"<<endl;
  int *iparam;
  iparam = new int[11];
  iparam[0] = 1;
  iparam[2] = 3*n;
  iparam[6] = 1;
  cout<<"ipntr"<<endl;
  int *ipntr;
  ipntr = new int[11];
  cout<<"workd"<<endl;
  double *workd;
  workd = new double[3*n];
  cout<<"workl"<<endl;
  double *workl;
  workl = new double[ncv*(ncv+8)];
  cout<<"lworkl"<<endl;
  int lworkl = ncv*(ncv+8);
  int info = 0;
  int rvec = 1;  // Changed from above
  cout<<"select"<<endl;
  int *select;
  select = new int[ncv];
  cout<<"d"<<endl;
  double *d;
  d = new double[2*ncv];
  double sigma;
  int ierr;

  cout<<"main loop"<<endl;
  do {
	cout<<"calling dsaupd_"<<endl;
    dsaupd_(&ido, bmat, &n, which, &nev, &tol, resid, 
	    &ncv, v, &ldv, iparam, ipntr, workd, workl,
	    &lworkl, &info);
    cout<<"successful call"<<endl;
    if ((ido==1)||(ido==-1)) av(n, workd+ipntr[0]-1, workd+ipntr[1]-1);
  } while ((ido==1)||(ido==-1));

  if (info<0) {
         cout << "Error with dsaupd, info = " << info << "\n";
         cout << "Check documentation in dsaupd\n\n";
  } else {
    dseupd_(&rvec, "All", select, d, v, &ldv, &sigma, bmat,
	    &n, which, &nev, &tol, resid, &ncv, v, &ldv,
	    iparam, ipntr, workd, workl, &lworkl, &ierr);

    if (ierr!=0) {
      cout << "Error with dseupd, info = " << ierr << "\n";
      cout << "Check the documentation of dseupd.\n\n";
    } else if (info==1) {
      cout << "Maximum number of iterations reached.\n\n";
    } else if (info==3) {
      cout << "No shifts could be applied during implicit\n";
      cout << "Arnoldi update, try increasing NCV.\n\n";
    }

    int i, j;
    for (i=0; i<nev; i++) Evals[i] = d[i];
    for (i=0; i<nev; i++) for (j=0; j<n; j++) Evecs[j][i] = v[i*n+j];

    delete resid;
    delete v;
    delete iparam;
    delete ipntr;
    delete workd;
    delete workl;
    delete select;
    delete d;
  }
}

DisjointSetForest normalizedCuts(WeightedGraph &graph, double stop) {
	cout<<"computing normalized laplacian"<<endl;
	bigMatrix = normalizedSparseLaplacian(graph);
	cout<<"initializing eigenvalue and eigenvectors storage"<<endl;
	double *evals = new double[2];
	double **evecs = new double*[2];

	for (int i = 0; i < 2; i++) {
		evecs[i] = new double[graph.numberOfVertices()];
	}

	cout<<"computing smallest eigenvalues and corresponding eigenvectors"<<endl;
	dsaupd(graph.numberOfVertices(), 2, evals, evecs);

	cout<<"eigenvalue: "<<evals[1]<<endl;
	cout<<"eigenvector: ";

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		cout<<evecs[1][i]<<", ";
	}

	cout<<endl;
	
	delete[] evals;
	for (int i = 0; i < 2; i++) {
		delete[] evecs[i];
	}
	delete[] evecs;

	return DisjointSetForest(0);
}
