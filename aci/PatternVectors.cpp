#include "PatternVectors.h"

VectorXd evaluatePowerSymmetricPolynomials(const VectorXd &inputs) {
	VectorXd terms = VectorXd::Ones(inputs.size());
	VectorXd results(inputs.size());

	// accumulates terms for each polynomials for efficiency
	for (int i = 0; i < inputs.size(); i++) {
		terms = terms.cwiseProduct(inputs);

		results(i) = terms.sum();
	}

	return results;
}

VectorXd evaluateSymmetricPolynomials(const VectorXd &inputs) {
	// first evaluate all power symmetric polynomials
	VectorXd P = evaluatePowerSymmetricPolynomials(inputs);
	// then recursively compute the elementary symmetric polynomials
	VectorXd S(inputs.size() + 1); 
	// the extra element contains S[0] = 1 which we discard at the end
	S(0) = 1;
	S(1) = P(0);
	double outerSign = -1;

	for (int r = 2; r <= inputs.size(); r++) {
		double innerSign = r % 2 == 0 ? -1 : 1;
		double sum = 0;
		
		for (int k = 1; k <= r; k++) {
			sum += innerSign * P(k-1) * S(r - k);
			innerSign *= -1;
		}

		S(r) = (outerSign/(double)r) * sum;
	}

	return S.tail(inputs.size()); // we drop S(0)
}

static double signum(double x) {
	return (double)((x > 0) - (x < 0));
}

vector<VectorXd> patternVectors(vector<WeightedGraph> &graphs, int k, int maxGraphSize) {
	vector<VectorXd> vectors;
	vectors.reserve(graphs.size());

	for (int i = 0; i < (int)graphs.size(); i++) {
		// compute Laplacian matrix of the graph, padded with 0 to
		// the largest graph size
		MatrixXd laplacian = eigLaplacian(graphs[i]);
		MatrixXd padded = MatrixXd::Zero(maxGraphSize, maxGraphSize);

		padded.block(0, 0, graphs[i].numberOfVertices(), graphs[i].numberOfVertices())
			= laplacian;

		// compute eigenvectors and eigenvalues of the laplacian
		SelfAdjointEigenSolver<MatrixXd> eigenSolver(padded);

		// only take the k smallest eigenvalues/vectors which were
		// not introduced by padding. Padding introduces additional
		// isolated vertices, thereby artificially increasing the 
		// multiplicity of the eigenvalue 0.
		int paddingDifference = maxGraphSize - graphs[i].numberOfVertices() + 1;

		VectorXd eigenvalues = eigenSolver.eigenvalues().head(k + paddingDifference).tail(k);
		MatrixXd eigenvectors = eigenSolver.eigenvectors().block(0, paddingDifference, maxGraphSize, k);

		//cout<<"eigenvalues: "<<eigenvalues.transpose()<<endl;
		//cout<<"eigenvectors: "<<eigenvectors<<endl;

		MatrixXd S(maxGraphSize, k);

		// compute columns of the spectral matrix, and evaluate the 
		// symmetric polynomials on each of them
		for (int j = 0; j < k; j++) {
			// additional condition in case the eigensolver returns some
			// negative close to 0 value.
			VectorXd spectralMatrixCol = (eigenvalues(j) > 0 ? sqrt(eigenvalues(j)) : 0) * eigenvectors.col(j);

			S.col(j) = evaluateSymmetricPolynomials(spectralMatrixCol);
			//cout<<"eigenvalue "<<eigenvalues(j)<<endl;
			//cout<<"eigenvector "<<eigenvectors.col(j).transpose()<<endl;
			//cout<<"spectral matrix col: "<<endl<<spectralMatrixCol.transpose()<<endl;
			//cout<<"evaluates to:"<<endl<<S.col(j).transpose()<<endl;
		}
		// form pattern vector from the results of the symmetric 
		// polynomials by applying logarithmic scaling.
		VectorXd F(maxGraphSize * k);

		for (int i = 0; i < maxGraphSize; i++) {
			for (int j = 0; j < k; j++) {
				F(toRowMajor(k, j, i)) = signum(S(i,j)) * log(1 + abs(S(i,j)));
			}
		}

		vectors.push_back(F);
	}

	return vectors;
}
