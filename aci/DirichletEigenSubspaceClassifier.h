#pragma once

#include <vector>
#include <tuple>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <iostream>
#include <fstream>

#include "GraphSpectra.h"
#include "SubspaceComparison.h"
#include "CSVIterator.h"
#include "DisjointSet.hpp"

using namespace std;
using namespace Eigen;

/**
 * Classifies graphs by measuring similarity between subspaces spanned by the
 * smallest Dirichlet eigenvectors of some induced subgraph "centered" around
 * a "face" vertex.
 *
 * The notion of "face" vertex relates to the case of animation character
 * classification, where we expect the face to always be in the character image,
 * and the vertex should correspond to some face segment.
 *
 * The induced subgraph is "centered" around said vertex as a heuristic for
 * efficiency - we cannot enumerate all induced subgraphs efficiently, and it
 * has been shown that determining the largest common (induced) subgraph is a
 * NP-hard problem. So we heuristically take subgraphs rooted around this vertex,
 * using BFS.
 */
class DirichletEigenSubspaceClassifier {
public:
	enum SimilarityType { AVERAGE, SMALLEST };
private:
	// training base of matrix graph representations with associated class label
	vector<pair<MatrixXd, int> > trainingBase;
	int nbEigenvectors;
	double minSubgraphSize;
	SimilarityType simType;

	/**
	 * Computes the normalized Laplcian of a graph, with vertices permuted
	 * by BFS order starting from a specific face vertex.
	 *
	 * @param graph graph to compute the normalized Laplacian from.
	 * @param faceVertex vertex to start BFS computation from.
	 * @param simType type of similarity to compute from canonical angles,
	 * either average canonical angle or smallest canonical angle.
	 * @return normalized Laplacian of the graph, with rows and columns indexed
	 * by vertices in BFS order.
	 */
	MatrixXd bfsNormalizedLaplacian(const WeightedGraph &graph, int faceVertex);

	/**
	 * Compute the similarity between graphs represented by their Laplacian
	 * matrix permuted in BFS order from the face vertex by measuring the
	 * canonical angles between subspaces spanned by the Dirichlet eigenvectors
	 * of select induced subgraphs. Now that was a mouthful.
	 *
	 * @param l1 Laplacian matrix of the first graph.
	 * @param l2 Laplacian matrix of the second graph.
	 * @return a measure of similarity between 0 and 1 between the 2 graphs. Magic.
	 */
	double computeSimilarity(const MatrixXd &l1, const MatrixXd &l2);

public:
	/**
	 * Initializes the classifier with a given number of smallest eigenvectors
	 * to compute, as well as a minimum induced subgraph size to consider as a
	 * a ratio of the smallest graph of the comparison.
	 *
	 * @param nbEigenvectors number of eigenvectors, greater or equal to 1.
	 * @param minSubgraphSize minimum subgraph size to consider, in the ]0;1[
	 * interval. Ratio of the number of vertices of the smallest graph involved
	 * in each comparison, while noting that we will always ignore at least 1
	 * vertex for each graph. This is because most theoretical results on induced
	 * subgraphs with Dirichlet boundary condition assume a non-empty boundary.
	 */
	DirichletEigenSubspaceClassifier(int nbEigenvectors, double minSubgraphSize, SimilarityType simType);

	/**
	 * Trains the classifier.
	 *
	 * @param trainingSamples training samples, each element being a triplet
	 * (G, l, f) where G is a graph, l is the class label associated to the graph
	 * and f is the face vertex of the graph.
	 */
	void train(vector<std::tuple<WeightedGraph, int, int> > trainingSamples);

	/**
	 * Predicts the class label of a graph with face vertex. Must be called
	 * after the train function has been called on some training samples.
	 *
	 * @param graph graph to predict the class label of.
	 * @param faceVertex face vertex of the graph.
	 * @return the expected class label of the graph.
	 */
	int predict(const WeightedGraph &graph, int faceVertex);
};

/**
 * Loads precomputed face positioins from a csv file, associated to the
 * corresponding image name.
 *
 * @param filename filename of the csv file to load.
 * @return a vector of pairs (name_i,p_i) where name_i is the name of the image
 * and p_i are the coordinates of the face in the image.
 */
vector<pair<string,Vector2d>, aligned_allocator<pair<string,Vector2d> > > loadFacePositions(string filename);

/**
 * computes the face segment as the one containing the face
 * position. To do this somewhat efficiently, need to compute the
 * mapping from root to linear index in the graph, and then simply
 * take the root by find operation on the disjoint set data structure
 * - modulo a row major transform.
 *
 * @param width width of the image (for row major transform)
 * @param segmentation segmentation of the image.
 * @param facePosition position of the face in the image.
 * @return the index of the segment containing the face position.
 */
int faceSegment(int width, DisjointSetForest &segmentation, Vector2d facePosition);
