#include "MultipleGraphsClassifier.h"

vector<VectorXd> averageColorLabeling(DisjointSetForest &segmentation, const Mat_<Vec3b> &image, const Mat_<float> &mask) {
	vector<VectorXd> averageColor;
	averageColor.reserve(segmentation.getNumberOfComponents());

	for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
		VectorXd zeros = VectorXd::Zero(3);
		averageColor.push_back(zeros);
	}
	map<int,int> rootIndexes = segmentation.getRootIndexes();

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (mask(i,j) > 0) {
				int root = segmentation.find(toRowMajor(image.cols, j, i));
				int segmentIndex = rootIndexes[root];
				VectorXd pixColor(3);
				pixColor(0) = image(i,j)[0];
				pixColor(1) = image(i,j)[1];
				pixColor(2) = image(i,j)[2];

				averageColor[segmentIndex] += pixColor / (float)segmentation.getComponentSize(root);
			}
		}
	}

	return averageColor;
}

vector<VectorXd> gravityCenterLabeling(DisjointSetForest &segmentation, const Mat_<Vec3b> &image, const Mat_<float> &mask) {
	vector<Vec2f> centers;
	gravityCenters(image, mask, segmentation, centers);

	vector<VectorXd> eigCenters;
	eigCenters.reserve(centers.size());

	for (int i = 0; i < (int)centers.size(); i++) {
		VectorXd center(2);

		center(0) = centers[i](0) / (double)image.rows;
		center(1) = centers[i](1) / (double)image.cols;

		eigCenters.push_back(center);
	}

	return eigCenters;
}

MultipleGraphsClassifier::MultipleGraphsClassifier(vector<std::tuple<SegmentLabeling,double> > features, int k) 
	: features(features), k(k)
{
}

WeightedGraph MultipleGraphsClassifier::computeFeatureGraph(int feature, DisjointSetForest &segmentation, const Mat_<Vec3b> &image, const Mat_<float> &mask) {
	vector<VectorXd> featureVectors = get<0>(this->features[feature])(segmentation, image, mask);
	MatrixXd similarityMatrix = MatrixXd::Zero(segmentation.getNumberOfComponents(), segmentation.getNumberOfComponents());

	// compute the similarity matrix
	for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
		for (int j = 0; j < segmentation.getNumberOfComponents(); j++) {
			similarityMatrix(i,j) = exp(-(featureVectors[i] - featureVectors[j]).squaredNorm() / pow(get<1>(this->features[feature]), 2));
		}
	}

	// compute k nearest neighbor graph from similarity matrix
	KNearestGraph kNearest(min(20, segmentation.getNumberOfComponents()));
	WeightedGraph featureGraph;
	DenseSimilarityMatrix denseSimMat(&similarityMatrix);

	kNearest(denseSimMat, featureGraph);

	return featureGraph;
}

static bool compareSampleSize(const std::tuple<DisjointSetForest, Mat_<Vec3b>, Mat_<float>, int > &g1, const std::tuple<DisjointSetForest, Mat_<Vec3b>, Mat_<float>, int > &g2) {
	return get<0>(g1).getNumberOfComponents() < get<0>(g2).getNumberOfComponents();
}

/**
 * The training procedure computes and store the graphs for each training
 * sample. Clears any previous training data. Graphs are stored in BFS order
 * starting from the face vertex.
 */
void MultipleGraphsClassifier::train(vector<std::tuple<DisjointSetForest, Mat_<Vec3b>, Mat_<float>, int > > trainingSet) {
	this->maxTrainingGraphSize = get<0>(*max_element(trainingSet.begin(), trainingSet.end(), compareSampleSize)).getNumberOfComponents();
	this->minTrainingGraphSize = get<0>(*min_element(trainingSet.begin(), trainingSet.end(), compareSampleSize)).getNumberOfComponents();
	this->trainingFeatureGraphs.clear();
	this->trainingFeatureGraphs.reserve(trainingSet.size());

	for (int i = 0; i < (int)trainingSet.size(); i++) {
		DisjointSetForest segmentation;
		Mat_<Vec3b> image;
		Mat_<float> mask;
		int label;
		std::tie (segmentation, image, mask, label) = trainingSet[i];

		vector<WeightedGraph> featureGraphs;
		featureGraphs.reserve(this->features.size());

		for (int i = 0; i < (int)this->features.size(); i++) {
			featureGraphs.push_back(this->computeFeatureGraph(i, segmentation, image, mask));
		}

		this->trainingFeatureGraphs.push_back(std::tuple<vector<WeightedGraph>, int >(featureGraphs, label));
	}
}

int MultipleGraphsClassifier::predict(DisjointSetForest &segmentation, const Mat_<Vec3b> &image, const Mat_<float> &mask) {
	int maxGraphSize = max(segmentation.getNumberOfComponents(), this->maxTrainingGraphSize);
	int minGraphSize = min(segmentation.getNumberOfComponents(), this->minTrainingGraphSize);

	if (minGraphSize <= this->k) {
		cout<<"the smallest graph has size "<<minGraphSize<<", cannot compute "<<this->k<<" eigenvectors"<<endl;
		exit(EXIT_FAILURE);
	}

	// compute each feature graph for the test sample
	vector<WeightedGraph> featureGraphs;
	featureGraphs.reserve(this->features.size());

	for (int i = 0; i < (int)this->features.size(); i++) {
		featureGraphs.push_back(this->computeFeatureGraph(i, segmentation, image, mask));
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

	// concatenate pattern vectors by image, converting to opencv format
	// to work with cv's ML module.
	Mat_<float> longPatterns = Mat_<float>::zeros(this->trainingFeatureGraphs.size() + 1, maxGraphSize * k * this->features.size());

	for (int i = 0; i < (int)this->trainingFeatureGraphs.size() + 1; i++) {
		VectorXd longPattern(maxGraphSize * k * this->features.size());
		for (int j = 0; j < this->features.size(); j++) {
			longPattern.block(j * maxGraphSize * k, 0, maxGraphSize * k, 1) = patternsByFeatures[j][i];
		}

		Mat_<double> cvLongPattern;

		eigenToCv(longPattern, cvLongPattern);

		Mat_<float> castLongPattern = Mat_<float>(cvLongPattern);

		castLongPattern.copyTo(longPatterns.row(i));
	}

	// classification of long patterns using SVM
	CvKNearest svmClassifier;

	Mat_<int> classes(this->trainingFeatureGraphs.size(), 1);
	
	for (int i = 0; i < (int)this->trainingFeatureGraphs.size(); i++) {
		classes(i,0) = get<1>(this->trainingFeatureGraphs[i]);
	}

	svmClassifier.train(longPatterns.rowRange(1, longPatterns.rows), classes);

	return (int)svmClassifier.find_nearest(longPatterns.row(0), 1);
}
