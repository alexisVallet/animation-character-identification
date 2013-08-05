#pragma once

#include "GraphSpectra.h"
#include "DisjointSet.hpp"
#include "SpectralClustering.h"
#include "PatternVectors.h"

#include <tuple>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

/**
 * Datatype for segment labelling function associating vectors
 * to each segment.
 */
typedef vector<VectorXd> (*SegmentLabeling)(DisjointSetForest &segmentation, const Mat_<Vec3b> &image, const Mat_<float> &mask);

/**
 * Classifier computing one graph for each feature of interest in the
 * animation image - for instance one for position information, one for
 * color informatin, another for shape information.
 */
class MultipleGraphsClassifier {
private:
	vector<std::tuple<SegmentLabeling, double> > features;
	int k;
	vector<std::tuple<vector<WeightedGraph>, int > > trainingFeatureGraphs;
	int maxTrainingGraphSize;

	/**
	 * Computes a feature graph of a segmented image.
	 *
	 * @param feature index of the segment labeling function to use as
	 * feature space.
	 * @param segmentation segmentation of the image.
	 * @param image segmented image.
	 * @param mask mask of the image.
	 * @return the corresponding feature graph.
	 */
	WeightedGraph computeFeatureGraph(int feature, DisjointSetForest &segmentation, const Mat_<Vec3b> &image, const Mat_<float> &mask, int faceSegment);

public:
	/**
	 * Initializes the classifier with a set of features to characterize
	 * segments with, and the type of similarity between graphs to use.
	 *
	 * @param features features to compute from each image segment, with
	 * associated scale parameter.
	 * @param simType type of similarity function to use to compare graph,
	 * can be:
	 * - CANONICAL_ANGLES_MIN measures similarity using the cosine of the
	 * smallest canonical angle between the subspaces spanned by the k
	 * smallest eigenvectors of subgraphs.
	 * - CANONICAL_ANGLES_AVG measures similarity using the average canonical
	 * angle between subspaces.
	 * - PATTERN_VECTOR measures similarity by euclid distance between
	 * pattern vectors computed using symmetric polynomials, as defined by
	 * Hancock, Luo, Wilson, 2005.
	 * @param k the maximum number of graph eigenvectors to use for any
	 * of the similarity functions.
	 */
	MultipleGraphsClassifier(vector<std::tuple<SegmentLabeling, double> > features, int k);
	/**
	 * Trains the classifier using a set of segmented images, with 
	 * corresponding masks and class labels.
	 *
	 * @param trainingSet training set to train the classifier with. Each
	 * element of the vector is a tuple (S, I, M, f, l) where S is a segmentation
	 * for image I with mask M, f is the face segment and l is integer class label of the sample.
	 */
	void train(vector<std::tuple<DisjointSetForest, Mat_<Vec3b>, Mat_<float>, int, int > > trainingSet);

	/**
	 * Predicts the class of a segmented image from previous training
	 * information.
	 *
	 * @param segmentation segmentation of the image.
	 * @param image segmented image.
	 * @param mask mask of the image.
	 * @return the predicted class label of the test sample.
	 */
	int predict(DisjointSetForest &segmentation, const Mat_<Vec3b> &image, const Mat_<float> &mask, int faceSegment);
};
