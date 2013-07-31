#include "SpectralClusteringSegmentation.h"

PixelSimilarityMatrix::PixelSimilarityMatrix(const Mat_<Vec3b> &image, double colorSigma, double positionSigma)
	: image(image), colorSigma(colorSigma), positionSigma(positionSigma)
{

}

double PixelSimilarityMatrix::operator() (int i, int j) const {
	cout<<"computing similarity between "<<i<<" and "<<j<<endl;
	pair<int,int> iposition = fromRowMajor(this->image.cols, i);
	pair<int,int> jposition = fromRowMajor(this->image.cols, j);
	Vec3b icolor = this->image(iposition.first, iposition.second);
	Vec3b jcolor = this->image(jposition.first, jposition.second);

	cout<<"jposition = "<<jposition.first<<", "<<jposition.second<<endl;
	cout<<"iposition = "<<iposition.first<<", "<<iposition.second<<endl;
	cout<<"icolor = "<<icolor<<endl;
	cout<<"jcolor = "<<jcolor<<endl;

	return
		exp(- pow(norm(icolor - jcolor), 2) / pow(this->colorSigma, 2)) *
		exp(- (pow((double)iposition.first - jposition.first, 2) + pow((double)iposition.second - jposition.second, 2))/pow(this->positionSigma, 2));
}

int PixelSimilarityMatrix::rows() const {
	return this->image.rows * this->image.cols;
}

int PixelSimilarityMatrix::cols() const {
	return this->image.rows * this->image.cols;
}



DisjointSetForest spectralClusteringSegmentation(const Mat_<Vec3b> &image, const Mat_<float> &mask, int nbSegments) {
	// compute linear mask
	cout<<"computing linear mask"<<endl;
	vector<bool> linMask(mask.rows * mask.cols, false);

	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			linMask[toRowMajor(mask.cols, j, i)] = mask(i,j) > 0;
		}
	}

	PixelSimilarityMatrix internalMat(image, 147, sqrt(pow((double)mask.rows, 2) + pow((double)mask.cols, 2))/3);
	KNearestGraph internalGraph(8);
	MaskedGraph masked(linMask, &internalGraph);

	VectorXi classLabels;

	cout<<"running spectral clustering"<<endl;
	spectralClustering(internalMat, masked, _normalizedSparseLaplacian, nbSegments, classLabels, true, true);

	// convert the class labels into disjoint set data structure for convenience
	cout<<"converting class labels"<<endl;
	DisjointSetForest segmentation(classLabels.size());

	vector<int> classRep(nbSegments, -1);

	for (int i = 0; i < (int)classLabels.size(); i++) {
		if (classRep[classLabels[i]] < 0) {
			classRep[classLabels[i]] = i;
		} else {
			segmentation.setUnion(classRep[classLabels[i]], i);
		}
	}

	return segmentation;
}
