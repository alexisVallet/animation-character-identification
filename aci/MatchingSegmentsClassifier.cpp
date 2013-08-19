#include "MatchingSegmentsClassifier.h"

static bool compareSim(const std::tuple<int, int, double> &s1, const std::tuple<int, int, double> &s2) {
	return get<2>(s1) > get<2>(s2);
}

vector<std::tuple<int, int, double> > mostSimilarSegments(
	DisjointSetForest &lSeg, const Mat_<Vec3b> &lImage, const Mat_<float> &lMask,
	DisjointSetForest &sSeg, const Mat_<Vec3b> &sImage, const Mat_<float> &sMask,
	vector<pair<SegmentLabeling, double> > labelingFunctions,
	bool ignoreFirst) {
	int startSeg = ignoreFirst ? 1 : 0;
	// evaluate labeling functions on both segmentations
	vector<vector<VectorXd> > lLabels, sLabels;
	lLabels.reserve(labelingFunctions.size());
	sLabels.reserve(labelingFunctions.size());

	for (int i = 0; i < labelingFunctions.size(); i++) {
		lLabels.push_back(labelingFunctions[i].first(lSeg, lImage, lMask));
		sLabels.push_back(labelingFunctions[i].first(sSeg, sImage, sMask));
	}

	// evaluate similarity for all unordered pairs
	vector<std::tuple<int, int, double> > allPairsSimilarity;
	allPairsSimilarity.reserve((lSeg.getNumberOfComponents() - startSeg) * (sSeg.getNumberOfComponents() - startSeg));

	for (int i = startSeg; i < lSeg.getNumberOfComponents(); i++) {
		for (int j = startSeg; j < sSeg.getNumberOfComponents(); j++) {
			double sim = 1;

			for (int k = 0; k < labelingFunctions.size(); k++) {
				VectorXd l1 = lLabels[k][i];
				VectorXd l2 = sLabels[k][j];
				VectorXd diff = l1 - l2;

				sim *= exp(- diff.squaredNorm() / pow(labelingFunctions[k].second, 2));
			}

			allPairsSimilarity.push_back(std::tuple<int,int,double>(i,j,sim));
		}
	}

	// sort the pairs by similarity, and add them from most to least similar
	vector<std::tuple<int, int, double> > matching;
	matching.reserve((lSeg.getNumberOfComponents() - startSeg) * (sSeg.getNumberOfComponents() - startSeg));

	sort(allPairsSimilarity.begin(), allPairsSimilarity.end(), compareSim);

	for (int i = 0; i < allPairsSimilarity.size(); i++) {
		cout<<"similarity between "<<get<0>(allPairsSimilarity[i])<<" and "<<get<1>(allPairsSimilarity[i])<<" is "<<get<2>(allPairsSimilarity[i])<<endl;
		vector<Vec3b> colors1(lSeg.getNumberOfComponents(), Vec3b(0,0,0));
		vector<Vec3b> colors2(sSeg.getNumberOfComponents(), Vec3b(0,0,0));
		colors1[get<0>(allPairsSimilarity[i])] = Vec3b(255,255,255);
		colors2[get<1>(allPairsSimilarity[i])] = Vec3b(255,255,255);

		VectorXd avg1 = lLabels[0][get<0>(allPairsSimilarity[i])];
		VectorXd avg2 = sLabels[0][get<1>(allPairsSimilarity[i])];
		double a1 = lLabels[1][get<0>(allPairsSimilarity[i])](0);
		double a2 = sLabels[1][get<1>(allPairsSimilarity[i])](0);

		cout<<"a1 = "<<a1<<", a2 = "<<a2<<endl;
		cout<<"exp(-||avg1 - avg2||^2/sigma^2) = "<<exp(-(avg1-avg2).squaredNorm()/pow(labelingFunctions[0].second, 2))<<endl;
		cout<<"exp(-|a1 - a2|^2/phi^2) = "<<exp(-pow((a1 - a2) / labelingFunctions[1].second, 2))<<endl;

		imshow("s1", lSeg.toRegionImage(lImage, colors1));
		imshow("s2", sSeg.toRegionImage(sImage, colors2));
		waitKey(0);
	}

	vector<bool> lAdded(lSeg.getNumberOfComponents(), false);
	vector<bool> sAdded(sSeg.getNumberOfComponents(), false);

	for (int i = 0; i < allPairsSimilarity.size(); i++) {
		std::tuple<int,int,double> edge = allPairsSimilarity[i];

		if (!lAdded[get<0>(edge)] && !sAdded[get<1>(edge)]) {
			matching.push_back(edge);
			lAdded[get<0>(edge)] = true;
			sAdded[get<1>(edge)] = true;
		}
	}

	return matching;
}
