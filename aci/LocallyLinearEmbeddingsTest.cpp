#include "LocallyLinearEmbeddingsTest.h"

void testLocallyLinearEmbeddings() {
	Mat_<float> testSamples(20,5);
	Mat_<float> embeddings;

	randu(testSamples, -20, 20);

	locallyLinearEmbeddings(testSamples, 2, embeddings, 3);
}