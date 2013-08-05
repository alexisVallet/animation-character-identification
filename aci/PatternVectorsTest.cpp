#include "PatternVectorsTest.h"

#define TEST_VECTOR_SIZE 1000

/**
 * Checks permutation invariance on randomly generated data.
 */
static void testElementarySymmetricPolynomials() {
	VectorXd testInputs(TEST_VECTOR_SIZE);

	for (int i = 0; i < TEST_VECTOR_SIZE; i++) {
		testInputs(i) = (double)rand() / RAND_MAX;
	}

	VectorXd output = evaluateSymmetricPolynomials(testInputs);
	/*cout<<"normal input:"<<endl<<testInputs<<endl;
	cout<<"normal output:"<<endl<<output<<endl;*/
	random_shuffle(testInputs.data(), testInputs.data() + TEST_VECTOR_SIZE);
	VectorXd shuffleOutput = evaluateSymmetricPolynomials(testInputs);
	/*cout<<"shuffled input:"<<endl<<testInputs<<endl;
	cout<<"shuffled output:"<<endl<<shuffleOutput<<endl;*/

	assert(output.isApprox(shuffleOutput));
}

void testPatternVectors() {
	testElementarySymmetricPolynomials();
}
