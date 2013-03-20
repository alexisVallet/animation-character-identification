#include "Felzenszwalb.hpp"

static bool compareWeights(Edge edge1, Edge edge2) {
  return edge1.weight < edge2.weight;
}

DisjointSetForest felzenszwalbSegment(int k, WeightedGraph graph, int minCompSize) {
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

  // Post processing phase which fuses components below a certain size
  for (int i = 0; i < (int)edges.size(); i++) {
    int srcRoot = segmentation.find(edges[i].source);
    int dstRoot = segmentation.find(edges[i].destination);

    if (srcRoot != dstRoot && (segmentation.getComponentSize(srcRoot) <= minCompSize || segmentation.getComponentSize(dstRoot) <= minCompSize)) {
      segmentation.setUnion(srcRoot, dstRoot);
    }
  }

  return segmentation;
}

static float euclDist(Vec<uchar,3> v1, Vec<uchar,3> v2) {
  float dr = ((float)v1[0]) - ((float)v2[0]);
  float dg = ((float)v1[1]) - ((float)v2[1]);
  float db = ((float)v1[2]) - ((float)v2[2]);

  return sqrt(dr*dr + dg*dg + db*db);
}

WeightedGraph gridGraph(Mat_<Vec<uchar,3> > &image, ConnectivityType connectivity) {
  WeightedGraph grid(image.cols*image.rows, 4);
  // indicates neigbor positions depending on connectivity
  int numberOfNeighbors[2] = {2, 4};
  int colOffsets[2][4] = {{0, 1, 0, 0}, {-1, 0, 1, 1}};
  int rowOffsets[2][4] = {{1, 0, 0, 0}, { 1, 1, 1, 0}};

  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      int centerIndex = toRowMajor(image.cols, j,i);
      assert(centerIndex >= 0 && centerIndex < grid.numberOfVertices());
      Vec<uchar,3> centerIntensity = image(i,j);
      
      for (int n = 0; n < numberOfNeighbors[connectivity]; n++) {
	int neighborRow = i + rowOffsets[connectivity][n];
	int neighborCol = j + colOffsets[connectivity][n];
	
	if (neighborRow >= 0 && neighborRow < image.rows &&
	    neighborCol >= 0 && neighborCol < image.cols) {
	  int neighborIndex = toRowMajor(image.cols, neighborCol, neighborRow);
	  Vec<uchar,3> neighborIntensity = image(neighborRow, neighborCol);
	  
	  assert(neighborIndex >= 0 && neighborIndex < grid.numberOfVertices());
	  
	  grid.addEdge(centerIndex, neighborIndex, euclDist(centerIntensity, neighborIntensity));
	}
      }
    }
  }

  return grid;
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
