#include "SpectrumDistanceClustering.h"

SpectrumDistanceClustering::SpectrumDistanceClustering(SparseRepresentation matRep, bool symmetric, int k) 
	: matRep(matRep), symmetric(symmetric), k(k)
{

}

void SpectrumDistanceClustering::embed(const vector<WeightedGraph> &samples, MatrixXd &embeddings) {
	MatrixXd spectra(samples.size(), k + 2);

	for (int i = 0; i < (int)samples.size(); i++) {
		cout<<"computing spectrum of matrix "<<i<<endl;
		SparseMatrix<double> representation = this->matRep(samples[i], false);
		if (representation.rows() < this->k + 3) {
			cout<<"resizing"<<endl;
			vector<Triplet<double> > tripletList;
			tripletList.reserve(representation.nonZeros());
			
			for (int j = 0; j < representation.outerSize(); ++j) {
				for (SparseMatrix<double>::InnerIterator it(representation,j); it; ++it) {
					tripletList.push_back(Triplet<double>(it.row(), it.col(), it.value()));
				}
			}

			representation.resize(this->k + 3, this->k + 3);
			representation.setFromTriplets(tripletList.begin(), tripletList.end());
		}
		cout<<"representation computed"<<endl;
		VectorXd evalues;
		MatrixXd evectors;

		if (this->symmetric) {
			symmetricSparseEigenSolver(representation, "SM", this->k + 2, (int)samples.size(), evalues, evectors);
		} else {
			nonSymmetricSparseEigenSolver(representation, "SR", this->k + 2, (int)samples.size(), evalues, evectors);
		}

		spectra.row(i) = evalues;
		cout<<evalues<<endl;
		cout<<"eigenvalues computed"<<endl;
	}

	embeddings = spectra.block(0, 2, samples.size(), k);
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
