#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "WeightedGraph.hpp"
#include "DisjointSet.hpp"
#include "SegmentationGraphClustering.h"
#include "GraphPartitions.h"
#include "Kernels.h"
#include "GraphSpectra.h"

/**
 * Clusters segmentation graphs of images by clustering of their respective
 * spectra by the K-means algorithm.
 */
class SpectrumDistanceClustering {
private:
	const SparseRepresentation matRep;
	const bool symmetric;
	const int k;

public:
	/**
	 * Initializes the algorithm with a specific image dataSet and the number k 
	 * of smallest eigenvalues to take into account.
	 *
	 * @param simFunc similarity functions between the vertices of a segmentation
	 */
	SpectrumDistanceClustering(SparseRepresentation matRep, bool symmetric, int k);

	/**
	 * Clusters segmentation graphs of images by clustering of their respective
	 * spectra by the K-means algorithm.
	 *
	 * @param segmentations segmentations of images corresponding to the graphs.
	 * @param samples segmentation graphs to cluster.
	 * @param nbClasses number of classes to cluster the graphs into.
	 * @param classLabels class label for each graph, in the {0, 1, ..., nbClasses - 1} set.
	 */
	void cluster(vector<DisjointSetForest> &segmentations, const vector<WeightedGraph> &samples, int nbClasses, VectorXi &classLabels);
};
