#include "SubspaceComparison.h"

void subspacesRotation(const MatrixXd &A, const MatrixXd &B, MatrixXd &Q) {
	assert(A.rows() == B.rows() && A.cols() == B.cols());
	MatrixXd C = B.transpose() * A;

	JacobiSVD<MatrixXd> svdFactorizer(C);

	MatrixXd U = svdFactorizer.matrixU();
	MatrixXd V = svdFactorizer.matrixV();

	Q = U * V.transpose();
}

void nullSpacesIntersection(const MatrixXd &A, const MatrixXd &B, MatrixXd &Y) {
	assert(A.cols() == B.cols());
	JacobiSVD<MatrixXd> ASVD(A);
	MatrixXd Va = ASVD.matrixV();
	int r = ASVD.nonzeroSingularValues();
	int n = A.cols();

	if (r < n) {
		MatrixXd subVa = Va.block(0, r, Va.rows(), n - r);
		MatrixXd C = B * subVa;
		JacobiSVD<MatrixXd> CSVD(C);
		MatrixXd Vc = CSVD.matrixV();
		int q = CSVD.nonzeroSingularValues();

		if (q < n - r) {
			MatrixXd Y = subVa * Vc.block(0, q, Vc.rows(), n - r - q);
		}
	}

	// trivial intersection
	Y = MatrixXd();
}

void canonicalAngles(const MatrixXd &A, const MatrixXd &B, MatrixXd &U, MatrixXd &V, VectorXd &cosines) {
	int p = A.cols(), q = B.cols();

	// A must have higher dimension than B, if it is not the case just inverts
	// them.
	if (p < q) {
		canonicalAngles(B, A, V, U, cosines);
		return;
	}

	ColPivHouseholderQR<MatrixXd> AQR(A), BQR(B);
	MatrixXd Qa = AQR.matrixQ(), Qb = BQR.matrixQ();
	MatrixXd C = Qa.transpose() * Qb;

	JacobiSVD<MatrixXd> CSVD(C);
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