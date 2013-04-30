#include "Felzenszwalb.hpp"

static bool compareWeights(Edge edge1, Edge edge2) {
  return edge1.weight < edge2.weight;
}

DisjointSetForest felzenszwalbSegment(int k, WeightedGraph graph, int minCompSize, Mat_<float> mask) {
	// sorts edge in increasing weight order
	vector<Edge> edges = graph.getEdges();
	sort(edges.begin(), edges.end(), compareWeights);

	// initializes the disjoint set forest to keep track of components, as
	// well as structures to keep track of component size and internal
	// differences.
	DisjointSetForest segmentation(graph.numberOfVertices());
	vector<float> internalDifferences(graph.numberOfVertices(), 0);

	// Goes through the edges, and fuses vertices if they pass a check,
	// updating internal differences.
	for (int i = 0; i < (int)edges.size(); i++) {
		Edge currentEdge = edges[i];
		int root1 = segmentation.find(currentEdge.source);
		int root2 = segmentation.find(currentEdge.destination);
		float mInt = min(internalDifferences[root1] 
			+ ((float)k)/((float)segmentation.getComponentSize(root1)),
			internalDifferences[root2] 
		    + ((float)k)/((float)segmentation.getComponentSize(root2)));

		if (root1 != root2 && currentEdge.weight <= mInt) {
			int newRoot = segmentation.setUnion(root1,root2);
			internalDifferences[newRoot] = currentEdge.weight;
		}
	}

	segmentation.fuseSmallComponents(graph, minCompSize);

	bool firstBgPixelFound = false;
	int bgSegment;

	// Fuses background into a single, unconnected component
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (mask(i,j) < 0.5) {
				if (!firstBgPixelFound) {
					bgSegment = toRowMajor(mask.cols, j, i);
					firstBgPixelFound = true;
				} else {
					segmentation.setUnion(bgSegment, toRowMajor(mask.cols, j, i));
				}
			}
		}
	}

	return segmentation;
}

DisjointSetForest combineSegmentations(const WeightedGraph &graph, vector<DisjointSetForest> &segmentations) {
  DisjointSetForest combination(graph.numberOfVertices());
  vector<Edge> edges = graph.getEdges();

  for (int i = 0; i < (int)edges.size(); i++) {
    Edge edge = edges[i];
    bool areInSameComponents = true;

    for (int j = 0; j < (int)segmentations.size(); j++) {
      int sourceRoot = segmentations[j].find(edge.source);
      int destinationRoot = segmentations[j].find(edge.destination);
      
      areInSameComponents = 
	areInSameComponents && (sourceRoot == destinationRoot);
    }

    if (areInSameComponents) {
      combination.setUnion(edge.source, edge.destination);
    }
  }

  return combination;
}
