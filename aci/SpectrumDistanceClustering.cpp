#include "SpectrumDistanceClustering.h"

SpectrumDistanceClustering::SpectrumDistanceClustering(SparseRepresentation matRep, bool symmetric, int k) 
	: matRep(matRep), symmetric(symmetric), k(k)
{

}

void SpectrumDistanceClustering::embed(const vector<WeightedGraph> &samples, MatrixXd &embeddings) {
	MatrixXd spectra(samples.size(), k + 1);

	for (int i = 0; i < (int)samples.size(); i++) {
		SparseMatrix<double> representation = this->matRep(samples[i], false);
		if (representation.rows() < this->k + 2) {
			representation.resize(this->k + 2, this->k + 2);
		}
		VectorXd evalues;
		MatrixXd evectors;

		if (this->symmetric) {
			symmetricSparseEigenSolver(representation, "SM", this->k + 1, (int)samples.size(), evalues, evectors);
		} else {
			nonSymmetricSparseEigenSolver(representation, "SM", this->k + 1, (int)samples.size(), evalues, evectors);
		}

		spectra.row(i) = evalues;
	}

	embeddings = spectra.block(0, 1, samples.size(), k);
}

void SpectrumDistanceClustering::cluster(const vector<WeightedGraph> &samples, int nbClasses, VectorXi &classLabels) {
	// compute specrum for each graph
	MatrixXd spectra;

	this->embed(samples, spectra);

	// clustering of spectra using K-means
	Mat_<double> cvSpectra(samples.size(), k, spectra.data());
	Mat_<int> cvClassLabels;

	cvSpectra = cvSpectra.t();

	kmeans(cvSpectra, nbClasses, cvClassLabels, TermCriteria(), 5, KMEANS_RANDOM_CENTERS);

	classLabels = VectorXi(samples.size());

	for (int i = 0; i < samples.size(); i++) {
		classLabels(i) = cvClassLabels(i,0);
	}
}
