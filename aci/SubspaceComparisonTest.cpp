#include "SubspaceComparisonTest.h"

static void testSubspacesRotation() {
	typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatT;
	double AData[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8 
	};
	double BData[] = {
		1.2, 2.1,
		2.9, 4.3,
		5.2, 6.1,
		6.8, 8.1
	};

	Map<MatT> A(AData, 4, 2);
	Map<MatT> B(BData, 4, 2);

	MatrixXd Q;

	subspacesRotation(A, B, Q);

	double expectedQData[] = {
		0.9999, -0.0126,
		0.0126, 0.9999
	};
	Map<MatT> expectedQ(expectedQData, 2, 2);

	assert(Q.isApprox(expectedQ, 10E-4));
}

static void testNullSpacesIntersection() {
	typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatT;
	double AData[] = {
		1, -1, 1,
		1, -1, 1,
		1, -1, 1
	};
	double BData[] = {
		4, 2, 0,
		2, 1, 0,
		6, 3, 0
	};
	double expectedYData[] = {
		1,
		-2,
		-3
	};

	Map<MatT> A(AData, 3, 3);
	Map<MatT> B(BData, 3, 3);
	Map<MatT> expectedY(expectedYData, 3, 1);

	MatrixXd Y;

	nullSpacesIntersection(A, B, Y);

	// colinearity test through dot product
	assert(abs((Y.transpose() * expectedY)(0)) - (Y.norm() * expectedY.norm()) <= 10E-4);
}

void testCanonicalAngles() {
	typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatT;
	double AData[] = {
		1, 2,
		3, 4,
		5, 6
	};
	double BData[] = {
		1, 5,
		3, 7,
		5, -1
	};
	Map<MatT> A(AData, 3, 2);
	Map<MatT> B(BData, 3, 2);
	cout<<A<<endl;
	cout<<B<<endl;
	VectorXd expectedCos(2);
	expectedCos(0) = 1.000;
	expectedCos(1) = 0.856;
	MatrixXd U, V;
	VectorXd cosines;

	canonicalAngles(A, B, U, V, cosines);

	cout<<expectedCos<<endl;
	cout<<cosines<<endl;

	assert(cosines.isApprox(expectedCos, 10E-4));

	double IData[] = {
		1, 0, 0,
		0, 1, 0,
		0, 0, 1
	};
	double CData[] = {
		0.5, 12,
		38, 52,
		13, 40
	};
	Map<MatT> I(IData, 3, 3);
	Map<MatT> C(CData, 3, 2);
	cout<<I<<endl;
	cout<<C<<endl;
	
	canonicalAngles(I,C,U,V,cosines);

	cout<<cosines<<endl;
}

void testSubspaceComparison() {
	testSubspacesRotation();
	testNullSpacesIntersection();
	testCanonicalAngles();
}
