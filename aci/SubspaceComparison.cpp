#include "SubspaceComparison.h"

void subspacesRotation(const MatrixXd &A, const MatrixXd &B, MatrixXd &Q) {
	assert(A.rows() == B.rows() && A.cols() == B.cols());
	MatrixXd C = B.transpose() * A;

	JacobiSVD<MatrixXd> svdFactorizer(C, ComputeFullU | ComputeFullV);

	MatrixXd U = svdFactorizer.matrixU();
	MatrixXd V = svdFactorizer.matrixV();

	Q = U * V.transpose();
}

void nullSpacesIntersection(const MatrixXd &A, const MatrixXd &B, MatrixXd &Y) {
	assert(A.cols() == B.cols());
	JacobiSVD<MatrixXd> ASVD(A, ComputeFullV);
	MatrixXd Va = ASVD.matrixV();
	int n = A.cols();
	int r = approxNonzeros(ASVD.singularValues());

	if (r < n) {
		MatrixXd subVa = Va.block(0, r, Va.rows(), n - r);
		MatrixXd C = B * subVa;
		JacobiSVD<MatrixXd> CSVD(C, ComputeFullV);
		MatrixXd Vc = CSVD.matrixV();
		int q = approxNonzeros(CSVD.singularValues());

		if (q < n - r) {
			MatrixXd subVc = Vc.block(0, q, Vc.rows(), n - r - q);
			Y = subVa * subVc;
		} else {
			Y = MatrixXd();
		}
	} else {
		Y = MatrixXd();
	}
}

void canonicalAngles(const MatrixXd &A, const MatrixXd &B, MatrixXd &U, MatrixXd &V, VectorXd &cosines) {
	assert(A.rows() == B.rows());
	int p = A.cols(), q = B.cols(), m = A.rows();

	// A must have higher dimension than B, if it is not the case just inverts
	// them.
	if (p < q) {
		canonicalAngles(B, A, V, U, cosines);
		return;
	}

	HouseholderQR<MatrixXd> AQR(A), BQR(B);
	MatrixXd Qa1 = AQR.householderQ();
	MatrixXd Qa = Qa1.block(0,0,m,p);
	MatrixXd Qb1 = BQR.householderQ();
	MatrixXd Qb = Qb1.block(0, 0, m, q);
	MatrixXd C = Qa.transpose() * Qb;

	JacobiSVD<MatrixXd> CSVD(C, ComputeFullU | ComputeFullV);
	MatrixXd Y = CSVD.matrixU(), Z = CSVD.matrixV();
	MatrixXd QaY = Qa * Y;

	U = QaY.block(0, 0, QaY.rows(), q);
	V = Qb * Z;
	cosines = CSVD.singularValues();
}

void subspacesIntersection(const MatrixXd &A, const MatrixXd &B, MatrixXd &C) {
	const double epsilon = 10E-8;
	MatrixXd U, V;
	VectorXd cosines;

	canonicalAngles(A, B, U, V, cosines);
	int s;
	for (s = 0; cosines(s) >= 1 - epsilon; s++);

	if (s == 0) {
		C = MatrixXd();
	} else {
		C = U.block(0, 0, U.rows(), s);
	}
}

double subspaceDistance(const MatrixXd &A, const MatrixXd &B) {
	MatrixXd U, V;
	VectorXd cosines;

	canonicalAngles(A, B, U, V, cosines);

	return sqrt(1 - pow(cosines(0), 2));
}