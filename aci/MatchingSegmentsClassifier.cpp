#include "MatchingSegmentsClassifier.h"

#define NB_FEATURES 3
#define COLOR_SIGMA 25
#define CENTERS_SIGMA 1
#define AREA_SIGMA 250

static bool compareSim(const std::tuple<int, int, double> &s1, const std::tuple<int, int, double> &s2) {
	return get<2>(s1) > get<2>(s2);
}

MatchingSegmentClassifier::MatchingSegmentClassifier(bool ignoreFirst) 
	: ignoreFirst(ignoreFirst), features(NB_FEATURES)
{
	// set up the fuzzy control system for segment similarity
	// a bit elaborate to get around the awkward and otherwise memory-leaky
	// API for fuzzylite.
	cout<<"preparing color input variable"<<endl;
	// average color input variable
	get<0>(features[0]) = averageColorLabeling;
	get<1>(features[0]) = COLOR_SIGMA;
	vector<fl::Triangle> colorTerms;
	colorTerms.push_back(fl::Triangle("LOW", 0, 0.5));
	colorTerms.push_back(fl::Triangle("MEDIUM", 0.25, 0.75));
	colorTerms.push_back(fl::Triangle("HIGH", 0.5, 1));
	get<3>(features[0]) = colorTerms;
	get<2>(features[0]).setName("Color");
	get<2>(features[0]).setRange(0,1);
	for (int i = 0; i < (int)colorTerms.size(); i++) {
		get<2>(features[0]).addTerm(&get<3>(features[0])[i]);
	}
	this->similarity.addInputVariable(&get<2>(features[0]));

	cout<<"preparing area input variable"<<endl;
	// segment area input variable
	get<0>(features[1]) = segmentAreaLabeling;
	get<1>(features[1]) = AREA_SIGMA;
	vector<fl::Triangle> areaTerms;
	areaTerms.push_back(fl::Triangle("LOW", 0, 0.5));
	areaTerms.push_back(fl::Triangle("MEDIUM", 0.25, 0.75));
	areaTerms.push_back(fl::Triangle("HIGH", 0.5, 1));
	get<3>(features[1]) = areaTerms;
	get<2>(features[1]).setName("Area");
	get<2>(features[1]).setRange(0,1);
	for (int i = 0; i < (int)areaTerms.size(); i++) {
		get<2>(features[1]).addTerm(&get<3>(features[1])[i]);
	}
	this->similarity.addInputVariable(&get<2>(features[1]));

	cout<<"preparing position input variable"<<endl;
	// gravity center input variable
	get<0>(features[2]) = gravityCenterLabeling;
	get<1>(features[2]) = CENTERS_SIGMA;
	vector<fl::Triangle> centerTerms;
	centerTerms.push_back(fl::Triangle("LOW", 0, 0.5));
	centerTerms.push_back(fl::Triangle("MEDIUM", 0.25, 0.75));
	centerTerms.push_back(fl::Triangle("HIGH", 0.5, 1));
	get<3>(features[2]) = centerTerms;
	get<2>(features[2]).setName("Position");
	get<2>(features[2]).setRange(0,1);
	for (int i = 0; i < (int)centerTerms.size(); i++) {
		get<2>(features[2]).addTerm(&get<3>(features[2])[i]);
	}
	this->similarity.addInputVariable(&get<2>(features[2]));

	cout<<"preparing similarity output variable"<<endl;
	// segment similarity output variable
	this->similarityOutput.setName("Similarity");
	this->similarityOutput.setRange(0,1);
	this->similarityOutput.setDefaultValue(fl::nan);
	this->similarityOutputTerms.push_back(fl::Triangle("LOW", 0, 0.4));
	this->similarityOutputTerms.push_back(fl::Triangle("MEDIUM", 0.2, 0.6));
	this->similarityOutputTerms.push_back(fl::Triangle("HIGH", 0.4, 0.8));
	this->similarityOutputTerms.push_back(fl::Triangle("VERYHIGH", 0.6, 1));
	for (int i = 0; i < (int)this->similarityOutputTerms.size(); i++) {
		this->similarityOutput.addTerm(&this->similarityOutputTerms[i]);
	}
	this->similarity.addOutputVariable(&this->similarityOutput);

	// ruleset
	cout<<"preparing ruleset"<<endl;
	string ruleStrings[] = {
		"if Color is HIGH and Area is HIGH and Position is HIGH then Similarity is VERYHIGH",
		"if Color is HIGH and Area is HIGH and Position is MEDIUM then Similarity is HIGH",
		"if Color is HIGH and Area is HIGH and Position is LOW then Similarity is HIGH",
		"if Color is MEDIUM and Area is HIGH then Similarity is MEDIUM",
		"if Area is MEDIUM and Color is HIGH then Similarity is MEDIUM",
		"if Area is MEDIUM and Color is MEDIUM then Similarity is LOW",
		"if Area is LOW or Color is LOW then Similarity is LOW",
		""
	};
	
	for (int i = 0; !ruleStrings[i].empty(); i++) {
		cout<<"adding rule "<<ruleStrings[i]<<endl;
		rules.addRule(fl::FuzzyRule::parse(ruleStrings[i], &this->similarity));
	}

	this->similarity.addRuleBlock(&rules);

	this->similarity.configure();

	string status;

	if (!this->similarity.isReady(&status)) {
		cout<<"could not initialize fuzzy engine:"<<endl<<status<<endl;
		exit(EXIT_FAILURE);
	}

	cout<<"successfullty initialized engine"<<endl;
}

vector<std::tuple<int, int, double> > MatchingSegmentClassifier::mostSimilarSegments(
	DisjointSetForest &lSeg, const Mat_<Vec3b> &lImage, const Mat_<float> &lMask,
	DisjointSetForest &sSeg, const Mat_<Vec3b> &sImage, const Mat_<float> &sMask) {
	int startSeg = ignoreFirst ? 1 : 0;
	// evaluate labeling functions on both segmentations
	vector<vector<VectorXd> > lLabels, sLabels;
	lLabels.reserve(features.size());
	sLabels.reserve(features.size());

	for (int i = 0; i < (int)features.size(); i++) {
		lLabels.push_back(get<0>(features[i])(lSeg, lImage, lMask));
		sLabels.push_back(get<0>(features[i])(sSeg, sImage, sMask));
	}

	// evaluate similarity for all pairs
	vector<std::tuple<int, int, double> > allPairsSimilarity;
	allPairsSimilarity.reserve((lSeg.getNumberOfComponents() - startSeg) * (sSeg.getNumberOfComponents() - startSeg));

	for (int i = startSeg; i < lSeg.getNumberOfComponents(); i++) {
		for (int j = startSeg; j < sSeg.getNumberOfComponents(); j++) {
			// evaluate similarity for each features individually
			for (int k = 0; k < (int)features.size(); k++) {
				VectorXd l1 = lLabels[k][i];
				VectorXd l2 = sLabels[k][j];
				VectorXd diff = l1 - l2;

				fl::scalar sim = exp(- diff.squaredNorm() / pow(get<1>(features[k]), 2));
				get<2>(this->features[k]).setInput(sim);
			}

			// run fuzzy similarity engine
			this->similarity.process();

			allPairsSimilarity.push_back(std::tuple<int,int,double>(i,j,this->similarityOutput.defuzzify()));
		}
	}

	// sort the pairs by similarity, and add them from most to least similar
	vector<std::tuple<int, int, double> > matching;
	matching.reserve((lSeg.getNumberOfComponents() - startSeg) * (sSeg.getNumberOfComponents() - startSeg));

	sort(allPairsSimilarity.begin(), allPairsSimilarity.end(), compareSim);

	/*for (int i = 0; i < allPairsSimilarity.size(); i++) {
		int seg1 = get<0>(allPairsSimilarity[i]);
		int seg2 = get<1>(allPairsSimilarity[i]);
		vector<Vec3b> colors1(lSeg.getNumberOfComponents(), Vec3b(0,0,0));
		vector<Vec3b> colors2(sSeg.getNumberOfComponents(), Vec3b(0,0,0));
		colors1[seg1] = Vec3b(255,255,255);
		colors2[seg2] = Vec3b(255,255,255);

		cout<<"similarity = "<<get<2>(allPairsSimilarity[i])<<endl;
		cout<<"color = "<<exp(-(lLabels[0][seg1] - sLabels[0][seg2]).squaredNorm() / pow(features[0].second, 2))<<endl;
		cout<<"area = "<<exp(-(lLabels[1][seg1] - sLabels[1][seg2]).squaredNorm() / pow(features[1].second, 2))<<endl;
		cout<<"position = "<<exp(-(lLabels[2][seg1] - sLabels[2][seg2]).squaredNorm() / pow(features[2].second, 2))<<endl;

		imshow("seg1", lSeg.toRegionImage(lImage, colors1));
		imshow("seg2", sSeg.toRegionImage(sImage, colors2));
		waitKey(0);
	}*/

	vector<bool> lAdded(lSeg.getNumberOfComponents(), false);
	vector<bool> sAdded(sSeg.getNumberOfComponents(), false);

	for (int i = 0; i < (int)allPairsSimilarity.size(); i++) {
		std::tuple<int,int,double> edge = allPairsSimilarity[i];

		if (!lAdded[get<0>(edge)] && !sAdded[get<1>(edge)]) {
			matching.push_back(edge);
			lAdded[get<0>(edge)] = true;
			sAdded[get<1>(edge)] = true;
		}
	}

	return matching;
}
