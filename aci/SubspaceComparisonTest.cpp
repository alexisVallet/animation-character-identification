#include "SubspaceComparisonTest.h"

static void testSubspacesRotation() {
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

	Map<MatrixXd> A = Map<MatrixXd>(AData, 4, 2);
	Map<MatrixXd> B = Map<MatrixXd>(BData, 4, 2);

	MatrixXd Q;

	subspacesRotation(A, B, Q);

	cout<<Q<<endl;
}

void testSubspaceComparison() {
	testSubspacesRotation();
}
