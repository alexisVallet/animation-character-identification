#include "MatchingSegmentsClassifier.h"

vector<pair<int, double> > mostSimilarSegments(
	DisjointSetForest &lSeg, const Mat_<Vec3b> &lImage, const Mat_<float> &lMask,
	DisjointSetForest &sSeg, const Mat_<Vec3b> &sImage, const Mat_<float> &sMask,
	vector<pair<SegmentLabeling, double> > labelingFunctions) {
	assert(lSeg.getNumberOfComponents() >= sSeg.getNumberOfComponents());

	// evaluate labeling functions on both segmentations
	vector<vector<VectorXd> > lLabels, sLabels;
	lLabels.reserve(labelingFunctions.size());
	sLabels.reserve(labelingFunctions.size());

	for (int i = 0; i < labelingFunctions.size(); i++) {
		lLabels.push_back(labelingFunctions[i].first(lSeg, lImage, lMask));
		sLabels.push_back(labelingFunctions[i].first(sSeg, sImage, sMask));
	}

	// Compute most similar segment in the smaller segmentation to each
	// segment in the largest.
	vector<pair<int, double> > matching;
	matching.reserve(lSeg.getNumberOfComponents());

	for (int i = 0; i < lSeg.getNumberOfComponents(); i++) {
		int maxIdx = 0;
		double maxSim = 0;

		for (int j = 0; j < sSeg.getNumberOfComponents(); j++) {
			double similarity = 1;

			for (int k = 0; k < labelingFunctions.size(); k++) {
				similarity *= exp(- (lLabels[k][i] - sLabels[k][j]).squaredNorm() / pow(labelingFunctions[k].second, 2));
			}

			if (maxSim <= similarity) {
				maxSim = similarity;
				maxIdx = j;
			}
		}

		matching.push_back(pair<int,double>(maxIdx,maxSim));
	}

	return matching;
}
