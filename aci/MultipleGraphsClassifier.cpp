#include "MultipleGraphsClassifier.h"

MultipleGraphsClassifier::MultipleGraphsClassifier(vector<std::tuple<SegmentLabeling,double> > features, int k) 
	: features(features), k(k)
{
	
}

WeightedGraph MultipleGraphsClassifier::computeFeatureGraph(int feature, DisjointSetForest &segmentation, const Mat_<Vec3b> &image, const Mat_<float> &mask, int faceSegment) {
	vector<VectorXd> featureVectors = get<0>(this->features[feature])(segmentation, image, mask);
	MatrixXd similarityMatrix = MatrixXd::Zero(segmentation.getNumberOfComponents(), segmentation.getNumberOfComponents());

	// compute the similarity matrix
	for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
		for (int j = 0; j < segmentation.getNumberOfComponents(); j++) {
			similarityMatrix(i,j) = exp(-(featureVectors[i] - featureVectors[j]).squaredNorm() / pow(get<1>(this->features[feature]), 2));
		}
	}

	// compute k nearest neighbor graph from similarity matrix
	KNearestGraph kNearest(min(8, segmentation.getNumberOfComponents()));
	WeightedGraph featureGraph;
	DenseSimilarityMatrix denseSimMat(&similarityMatrix);

	kNearest(denseSimMat, featureGraph);

	// permute vertices to get BFS order starting from face segment.
	vector<int> bfsOrder = breadthFirstSearch(featureGraph, faceSegment);
	return permuteVertices(featureGraph, bfsOrder);
}

static bool compareSampleSize(const std::tuple<DisjointSetForest, Mat_<Vec3b>, Mat_<float>, int, int > &g1, const std::tuple<DisjointSetForest, Mat_<Vec3b>, Mat_<float>, int, int > &g2) {
	return get<0>(g1).getNumberOfComponents() < get<0>(g2).getNumberOfComponents();
}

/**
 * The training procedure computes and store the graphs for each training
 * sample. Clears any previous training data. Graphs are stored in BFS order
 * starting from the face vertex.
 */
void MultipleGraphsClassifier::train(vector<std::tuple<DisjointSetForest, Mat_<Vec3b>, Mat_<float>, int, int > > trainingSet) {
	this->maxTrainingGraphSize = get<0>(*max_element(trainingSet.begin(), trainingSet.end(), compareSampleSize)).getNumberOfComponents();
	this->trainingFeatureGraphs.clear();
	this->trainingFeatureGraphs.reserve(trainingSet.size());

	for (int i = 0; i < (int)trainingSet.size(); i++) {
		DisjointSetForest segmentation;
		Mat_<Vec3b> image;
		Mat_<float> mask;
		int faceSegment;
		int label;
		std::tie (segmentation, image, mask, faceSegment, label) = trainingSet[i];

		vector<WeightedGraph> featureGraphs;
		featureGraphs.reserve(this->features.size());

		for (int i = 0; i < (int)this->features.size(); i++) {
			featureGraphs.push_back(this->computeFeatureGraph(i, segmentation, image, mask, faceSegment));
		}
	}
}

int MultipleGraphsClassifier::predict(DisjointSetForest &segmentation, const Mat_<Vec3b> &image, const Mat_<float> &mask, int faceSegment) {
	int maxGraphSize = max(segmentation.getNumberOfComponents(), this->maxTrainingGraphSize);
	// compute each feature graph for the test sample
	vector<WeightedGraph> featureGraphs;
	featureGraphs.reserve(this->features.size());

	for (int i = 0; i < (int)this->features.size(); i++) {
		featureGraphs.push_back(this->computeFeatureGraph(i, segmentation, image, mask, faceSegment));
	}

	// order graphs by feature, to compute pattern vectors feature by
	// feature.
	vector<vector<WeightedGraph> > graphsByFeatures(this->features.size());

	// add feature graphs for the test sample at index 0
	for (int i = 0; i < (int)this->features.size(); i++) {
		graphsByFeatures[i].reserve(this->trainingFeatureGraphs.size() + 1);
		graphsByFeatures[i].push_back(featureGraphs[i]);
	}

	// add feature graphs for each training sample
	for (int i = 0; i < (int)this->trainingFeatureGraphs.size(); i++) {
		for (int j = 0; j < (int)this->features.size(); j++) {
			graphsByFeatures[j].push_back(get<0>(this->trainingFeatureGraphs[i])[j]);
		}
	}

	// compute the corresponding pattern vectors
	vector<vector<VectorXd> > patternsByFeatures(this->features.size());
	
	for (int i = 0; i < (int)this->features.size(); i++) {
		patternsByFeatures[i] = patternVectors(graphsByFeatures[i], this->k, maxGraphSize);
	}

	// concatenate pattern vectors by image
	vector<VectorXd> longPatterns;
	longPatterns.reserve(this->trainingFeatureGraphs.size() + 1);

	for (int i = 0; i < (int)this->trainingFeatureGraphs.size() + 1; i++) {
		VectorXd longPattern(maxGraphSize * k * this->features.size());
		for (int j = 0; j < this->features.size(); j++) {
			longPattern.block(j * maxGraphSize * k, 0, maxGraphSize * k, 1) = patternsByFeatures[j][i];
		}
	}
}
