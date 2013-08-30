#include "MatchingSegmentsClassifier.h"

#define NB_FEATURES 3
#define COLOR_SIGMA 20
#define AREA_SIGMA 500
#define CENTERS_SIGMA 0.3

static bool compareSim(const std::tuple<int, int, double> &s1, const std::tuple<int, int, double> &s2) {
	return get<2>(s1) > get<2>(s2);
}

MatchingSegmentClassifier::MatchingSegmentClassifier(bool ignoreFirst) 
	: ignoreFirst(ignoreFirst), features(NB_FEATURES)
{
	// set up the fuzzy control system for segment similarity
	// a bit elaborate to get around the awkward API for fuzzylite.
	this->similarity = new fl::Engine("segment-similarity");

	//cout<<"preparing color input variable"<<endl;
	// average color input variable
	get<0>(features[0]) = averageColorLabeling;
	fl::InputVariable *color = new fl::InputVariable();
	color->setName("Color");
	color->setRange(0,1);
	color->addTerm(new fl::Triangle("LOW", 0, 0, 0.5));
	color->addTerm(new fl::Triangle("MEDIUM", 0, 0.5, 1));
	color->addTerm(new fl::Triangle("HIGH", 0.5, 1, 1));
	this->similarity->addInputVariable(color);
	get<1>(features[0]) = color;

	//cout<<"preparing area input variable"<<endl;
	// segment area input variable
	get<0>(features[1]) = segmentAreaLabeling;
	fl::InputVariable *area = new fl::InputVariable();
	area->setName("Area");
	area->setRange(0,1);
	area->addTerm(new fl::Triangle("LOW", 0, 0, 0.5));
	area->addTerm(new fl::Triangle("MEDIUM", 0, 0.5, 1));
	area->addTerm(new fl::Triangle("HIGH", 0.5, 1, 1));
	this->similarity->addInputVariable(area);
	get<1>(features[1]) = area;

	//cout<<"preparing position input variable"<<endl;
	// gravity center input variable
	get<0>(features[2]) = gravityCenterLabeling;
	fl::InputVariable *position = new fl::InputVariable();
	position->setName("Position");
	position->setRange(0,1);
	position->addTerm(new fl::Triangle("LOW", 0, 0, 0.5));
	position->addTerm(new fl::Triangle("MEDIUM", 0, 0.5, 1));
	position->addTerm(new fl::Triangle("HIGH", 0.5, 1, 1));
	this->similarity->addInputVariable(position);
	get<1>(features[2]) = position;

	//cout<<"preparing similarity output variable"<<endl;
	// segment similarity output variable
	this->similarityOutput = new fl::OutputVariable();
	this->similarityOutput->setName("Similarity");
	this->similarityOutput->setRange(0,1);
	this->similarityOutput->setDefaultValue(0);
	this->similarityOutput->addTerm(new fl::Triangle("LOW", 0, 0, 1./3.));
	this->similarityOutput->addTerm(new fl::Triangle("MEDIUM", 0, 1./3., 2./3.));
	this->similarityOutput->addTerm(new fl::Triangle("HIGH", 1./3., 2./3., 1));
	this->similarityOutput->addTerm(new fl::Triangle("VERYHIGH", 2./3., 1, 1));
	this->similarity->addOutputVariable(this->similarityOutput);

	// ruleset
	fl::RuleBlock *rules = new fl::RuleBlock();
	//cout<<"preparing ruleset"<<endl;
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
		//cout<<"adding rule "<<ruleStrings[i]<<endl;
		fl::MamdaniRule *rule = fl::MamdaniRule::parse(ruleStrings[i], this->similarity);
		//cout<<"rule parsed"<<endl;
		rules->addRule(rule);
		//cout<<"rule added"<<endl;
	}

	//cout<<"adding rule block"<<endl;
	this->similarity->addRuleBlock(rules);
	//cout<<"configuring"<<endl;
	this->similarity->configure("Minimum", "Maximum", "AlgebraicProduct", "AlgebraicSum", "Centroid");

	//cout<<"successfullty initialized engine"<<endl;
}

MatchingSegmentClassifier::~MatchingSegmentClassifier() {
	delete similarity;
}

void MatchingSegmentClassifier::computeSegmentLabels(DisjointSetForest &seg, const Mat_<Vec3f> &image, const Mat_<float> &mask, vector<vector<VectorXd> > &segmentLabels) {
	segmentLabels.clear();
	segmentLabels.reserve(this->features.size());

	for (int i = 0; i < (int)features.size(); i++) {
		segmentLabels.push_back(get<0>(features[i])(seg, image, mask));
	}
}

void MatchingSegmentClassifier::mostSimilarSegmentLabels(const vector<vector<VectorXd> > &lLabels, const vector<vector<VectorXd> > &sLabels, vector<std::tuple<int, int, double> > &matching, int lNbSeg, int sNbSeg) {
	int startSeg = ignoreFirst ? 1 : 0;

	// evaluate similarity for all pairs
	vector<std::tuple<int, int, double> > allPairsSimilarity;
	allPairsSimilarity.reserve((lNbSeg - startSeg) * (sNbSeg - startSeg));

	//cout<<"computing euclid distances"<<endl;
	vector<MatrixXd> euclidDistances(this->features.size());

	// first evaluate euclidean distances for automatic sigma determination
	for (int k = 0; k < (int)this->features.size(); k++) {
		euclidDistances[k] = MatrixXd::Zero(lNbSeg, sNbSeg);
		for (int i = startSeg; i < lNbSeg; i++) {
			for (int j = startSeg; j < sNbSeg; j++) {
				euclidDistances[k](i,j) = (lLabels[k][i] - sLabels[k][j]).norm();
			}
		}
	}

	//cout<<"computing variance for each feature"<<endl;
	// evaluate sigma^2 as the variance of the euclid distances for the feature
	vector<double> variances(this->features.size(), 0);

	for (int k = 0; k < (int)this->features.size(); k++) {
		double mean = euclidDistances[k].mean();

		for (int i = startSeg; i < lNbSeg; i++) {
			for (int j = startSeg; j < sNbSeg; j++) {
				variances[k] += pow(euclidDistances[k](i,j), 2);
			}
		}

		variances[k] = variances[k] / (double)((lNbSeg - startSeg) * (sNbSeg - startSeg));
		variances[k] -= pow(mean, 2);
	}

	//cout<<"computing similarity"<<endl;

	for (int i = startSeg; i < lNbSeg; i++) {
		for (int j = startSeg; j < sNbSeg; j++) {
			// evaluate similarity for each features individually
			for (int k = 0; k < (int)features.size(); k++) {
				fl::scalar sim = exp(- pow(euclidDistances[k](i,j), 2) / variances[k]);
				//cout<<"feature "<<k<<" has similarity "<<sim<<endl<<get<1>(this->features[k])->fuzzify(sim)<<endl;
				get<1>(this->features[k])->setInput(sim);
			}

			// run fuzzy similarity engine
			this->similarity->process();

			fl::scalar resultSimilarity = this->similarityOutput->defuzzify();

			//cout<<"similarity = "<<this->similarityOutput->fuzzify(resultSimilarity)<<endl;

			allPairsSimilarity.push_back(std::tuple<int,int,double>(i,j,resultSimilarity));
		}
	}

	// sort the pairs by similarity, and add them from most to least similar
	matching.clear();
	matching.reserve((lNbSeg - startSeg) * (sNbSeg - startSeg));

	sort(allPairsSimilarity.begin(), allPairsSimilarity.end(), compareSim);

	vector<bool> lAdded(lNbSeg, false);
	vector<bool> sAdded(sNbSeg, false);

	for (int i = 0; i < (int)allPairsSimilarity.size(); i++) {
		std::tuple<int,int,double> edge = allPairsSimilarity[i];

		if (!lAdded[get<0>(edge)] && !sAdded[get<1>(edge)]) {
			matching.push_back(edge);
			lAdded[get<0>(edge)] = true;
			sAdded[get<1>(edge)] = true;
		}
	}
}

vector<std::tuple<int, int, double> > MatchingSegmentClassifier::mostSimilarSegments(
	DisjointSetForest &lSeg, const Mat_<Vec3f> &lImage, const Mat_<float> &lMask,
	DisjointSetForest &sSeg, const Mat_<Vec3f> &sImage, const Mat_<float> &sMask) {
	
	// evaluate labeling functions on both segmentations
	vector<vector<VectorXd> > lLabels, sLabels;

	this->computeSegmentLabels(lSeg, lImage, lMask, lLabels);
	this->computeSegmentLabels(sSeg, sImage, sMask, sLabels);

	vector<std::tuple<int, int, double> > matching;

	this->mostSimilarSegmentLabels(lLabels, sLabels, matching, lSeg.getNumberOfComponents(), sSeg.getNumberOfComponents());

	return matching;
}

void MatchingSegmentClassifier::train(vector<std::tuple<DisjointSetForest, Mat_<Vec3f>, Mat_<float>, int> > &trainingSet) {
	this->trainingLabels.clear();
	this->trainingLabels.reserve(trainingSet.size());

	for (int i = 0; i < (int)trainingSet.size(); i++) {
		vector<vector<VectorXd> > segmentLabels; 
		this->computeSegmentLabels(get<0>(trainingSet[i]), get<1>(trainingSet[i]), get<2>(trainingSet[i]), segmentLabels);

		this->trainingLabels.push_back(std::tuple<vector<vector<VectorXd> >, int>(segmentLabels, get<3>(trainingSet[i])));
	}
}

int MatchingSegmentClassifier::predict(DisjointSetForest &segmentation, const Mat_<Vec3f> &image, const Mat_<float> &mask) {
	vector<vector<VectorXd> > segmentLabels;

	this->computeSegmentLabels(segmentation, image, mask, segmentLabels);
	map<int,int> rootIndexes = segmentation.getRootIndexes();
	vector<int> compSizes(segmentation.getNumberOfComponents(), 0);

	for (map<int,int>::iterator it = rootIndexes.begin(); it != rootIndexes.end(); it++) {
		compSizes[(*it).second] = segmentation.getComponentSize((*it).first);
	}

	int nearestNeighbor = 0;
	float maxSimilarity = 0;

	//cout<<"similarities: ";

	for (int i = 0; i < (int)this->trainingLabels.size(); i++) {
		vector<std::tuple<int, int, double> > matching;

		this->mostSimilarSegmentLabels(segmentLabels, get<0>(this->trainingLabels[i]), matching, segmentation.getNumberOfComponents(), get<0>(trainingLabels[i])[0].size());

		// compute weighted sum of matching similarities by size of test sample
		// segment area.
		double similarity = 0;

		for (int j = 0; j < (int)matching.size(); j++) {
			std::tuple<int,int,double> match = matching[j];

			similarity += compSizes[get<0>(match)] * get<2>(match);
		}

		//cout<<similarity<<", ";

		if (maxSimilarity < similarity) {
			maxSimilarity = similarity;
			nearestNeighbor = i;
		}
	}

	//cout<<endl;

	return get<1>(this->trainingLabels[nearestNeighbor]);
}
