/** @file */
#pragma once

#include "DisjointSet.hpp"
#include "WeightedGraph.hpp"
#include "LabeledGraph.hpp"

/**
 * Datatype describing graph labeling functions.
 */
typedef void (*Labeling)(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph);

/**
 * Base class for specifying vertex label and similariy function for a graph.
 */
class MatKernel {
public:
	virtual Labeling getLabeling() const = 0;
	virtual double operator() (const Mat&, const Mat&) const = 0;
};

/**
 * Fuses partitions of disjoint subgraphs of a graph G into a partition of G.
 *
 * @param graph the graph G containing all the subgraphs.
 * @param inSubgraph input vector indicating the partition yielding the subgraphs.
 * @param partitions input partitions to fuse.
 * @param partition output fused partition.
 */
void fusePartitions(const WeightedGraph &graph, vector<int> &inSubgraph, vector<int> &vertexIdx, vector<DisjointSetForest> &partitions, DisjointSetForest &partition);

/**
 * Removes isolated vertices from the graph, keeping track of which were removed
 * so they can be readded later. Typically used as a pre processing step to segmentation,
 * to remove isolated vertices from the background for instance.
 *
 * @param graph graph to remove isolated vertices from. Expects a bidirectional
 * representation of the graph.
 * @param vertexMap output vector associating to each vertex in the input vertex
 * the corresponding vertex in the output graph, -1 if the vertex is isolated in
 * the input graph.
 * @return subgraph of the input graph with no isolated vertices.
 */
WeightedGraph removeIsolatedVertices(WeightedGraph &graph, vector<int> &vertexMap);

/**
 * Adds isolated vertices which where previously removed from the graph by
 * removeIsolatedVertices. Typically used as a post processing step to segmentation
 * so the segmentation can be related to the pixels in the color image.
 *
 * @param graph graph to add isolated vertices to.
 * @param segmentation segmentation to compute the new segmentation from.
 * @param vertexMap input vector such as the one outputted by removeIsolatedVertices.
 * @return segmentation with additional isolated vertices added and fused into a single
 * background segment.
 */
DisjointSetForest addIsolatedVertices(WeightedGraph &graph, DisjointSetForest &segmentation, vector<int> &vertexMap);
