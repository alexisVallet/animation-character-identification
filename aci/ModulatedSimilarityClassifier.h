#pragma once

#include <vector>
#include <Eigen/Dense>
#include <boost/optional.hpp>
#include <iostream>

using namespace Eigen;
using namespace std;
using namespace boost;

class ModulatedSimilarityClassifier {
private:
	MatrixXd modulatedSimilarity;
	vector<int> classLabels;
	double magnification;
	double suppression;

public:
	ModulatedSimilarityClassifier(double magnification = 10, double suppression = 0.1);

	void train(const MatrixXd &similarity, const vector<int> &classLabels);

	void computeEmbedding(const VectorXd &testSimilarities, int dimension, MatrixXd &embedding) const;

	int predict(const VectorXd &testSimilarities, int dimension);
};
