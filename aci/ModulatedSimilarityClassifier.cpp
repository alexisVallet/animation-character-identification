#include "ModulatedSimilarityClassifier.h"

ModulatedSimilarityClassifier::ModulatedSimilarityClassifier(double magnification, double suppression)
	: magnification(magnification), suppression(suppression)
{
}

void ModulatedSimilarityClassifier::train(const MatrixXd &similarity, const vector<int> &classLabels) {
	assert(similarity.rows() == similarity.cols());
	assert(similarity.rows() == classLabels.size());
	this->modulatedSimilarity = MatrixXd(similarity);

	cout<<"computing modulated similarities"<<endl;
	for (int i = 0; i < (int) classLabels.size(); i++) {
		for (int j = i + 1; j < (int) classLabels.size(); j++) {
			if (classLabels[i] >= 0 && classLabels[j] >= 0) {
				if (classLabels[i] == classLabels[j]) {
					this->modulatedSimilarity(i,j) = this->magnification * this->modulatedSimilarity(i,j);
				} else {
					this->modulatedSimilarity(i,j) = this->suppression * this->modulatedSimilarity(i,j);
				}

				this->modulatedSimilarity(j,i) = this->modulatedSimilarity(i,j);
			}
		}
	}

	this->classLabels = vector<int>(classLabels);
}

void ModulatedSimilarityClassifier::computeEmbedding(const VectorXd &testSimilarities, int dimension, MatrixXd &embedding) const {
	assert(this->modulatedSimilarity.rows() == testSimilarities.size());
	assert(dimension < this->modulatedSimilarity.rows() -  1);

	cout<<"computing full similarity"<<endl;
	// compute graph laplacian
	MatrixXd fullSimilarities = MatrixXd::Zero(this->modulatedSimilarity.rows() + 1, this->modulatedSimilarity.cols() + 1);
	cout<<"copying training similarity"<<endl;
	fullSimilarities.block(0,0,this->modulatedSimilarity.rows(), this->modulatedSimilarity.cols()) = this->modulatedSimilarity;
	cout<<"copying test vector to the right"<<endl;
	fullSimilarities.block(0, this->modulatedSimilarity.rows(), this->modulatedSimilarity.rows(), 1) = testSimilarities;
	cout<<"copying test vector to the bottom"<<endl;
	MatrixXd testRow(testSimilarities);
	fullSimilarities.block(this->modulatedSimilarity.rows(), 0, 1, this->modulatedSimilarity.rows()) = testRow.transpose();

	cout<<"computing degrees"<<endl;
	VectorXd degrees(fullSimilarities.rows());

	for (int i = 0; i < fullSimilarities.rows(); i++) {
		degrees(i) = fullSimilarities.row(i).sum();
	}

	MatrixXd invD = MatrixXd::Zero(fullSimilarities.rows(), fullSimilarities.cols());
	cout<<"copying degrees"<<endl;
	for (int i = 0; i < fullSimilarities.rows(); i++) {
		invD(i,i) = degrees(i) < 0.0001 ? 0 : 1/degrees(i);
	}

	cout<<"computing eigenvectors of laplacian"<<endl;
	// compute the embedding by the dimension + 1 eigenvectors of the Laplacian
	EigenSolver<MatrixXd> solver(invD * fullSimilarities);

	cout<<(solver.info() == Eigen::Success ? "success" : "failure")<<endl;

	MatrixXd evectors = solver.eigenvectors().real();

	embedding = evectors.block(0, evectors.cols() - dimension - 2, evectors.rows(), dimension);
}

int ModulatedSimilarityClassifier::predict(const VectorXd &testSimilarities, int dimension) {
	MatrixXd embedding;

	cout<<"dimension = "<<dimension<<endl;
	cout<<"computing embedding"<<endl;
	this->computeEmbedding(testSimilarities, dimension, embedding);

	cout<<"computing nearest neighbor"<<endl;
	int nearestNeighbor = 0;
	double minDistance = DBL_MAX;
	VectorXd testCoord = embedding.row(embedding.rows() - 1);

	for (int i = 0; i < embedding.rows() - 1; i++) {
		VectorXd trainCoord = embedding.row(i);

		double distance = (testCoord - trainCoord).norm();

		if (distance < minDistance) {
			nearestNeighbor = i;
			minDistance = distance;
		}
	}

	return this->classLabels[nearestNeighbor];
}
