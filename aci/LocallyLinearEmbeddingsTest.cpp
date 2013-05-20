#include "LocallyLinearEmbeddingsTest.h"

static Vec3f swissRoll(float minRadius, float maxRadius, float x, float angle) {
	float radius = minRadius + (angle / (2 * M_PI)) * (maxRadius - minRadius);

	return Vec3f(x, radius * cos(angle), radius * sin(angle));
}

static void matToCsv(Mat_<float> mat, ostream &out) {
	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			out<<mat(i,j);
			if (j < mat.cols - 1) {
				out<<", ";
			}
		}
		out<<endl;
	}
}

void testLocallyLinearEmbeddings() {
	const int nbSamples = 1000;
	const float maxRadius = 50;
	const float minRadius = 20;
	const float length = 50;
	Mat_<float> testCoords(nbSamples, 2);
	Mat_<float> testSamples(nbSamples, 3);
	Mat_<float> embeddings;

	for (int i = 0; i < nbSamples; i++) {
		float randX = ((float)rand() / (float)RAND_MAX) * length;
		float randAngle = ((float)rand() / (float)RAND_MAX) * 2 * M_PI;

		testCoords(i,0) = randX;
		testCoords(i,1) = randAngle;

		Vec3f coords = swissRoll(minRadius, maxRadius, randX, randAngle);

		for (int j = 0; j < 3; j++) {
			testSamples(i,j) = coords(j);
		}
	}

	locallyLinearEmbeddings(testSamples, 2, embeddings, 6);

	ofstream coordsCsv, samplesCsv, embeddingsCsv;

	coordsCsv.open("../stats/swissRollParams.csv");
	samplesCsv.open("../stats/swissRoll3D.csv");
	embeddingsCsv.open("../stats/swissRoll2D.csv");

	matToCsv(testCoords, coordsCsv);
	matToCsv(testSamples, samplesCsv);
	matToCsv(embeddings, embeddingsCsv);
}