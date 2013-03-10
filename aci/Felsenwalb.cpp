#include "Felsenwalb.hpp"

static bool compareWeights(Edge edge1, Edge edge2) {
  return edge1.weight < edge2.weight;
}

DisjointSetForest felsenwalbSegment(int k, WeightedGraph graph) {
  vector<Edge> edges = graph.getEdges();
  cout<<"Graph has "<<graph.numberOfVertices()<<" vertices and "<<edges.size()<<" edges"<<endl;
  sort(edges.begin(), edges.end(), compareWeights);

  DisjointSetForest segmentation(graph.numberOfVertices());
  vector<float> internalDifferences(graph.numberOfVertices(), 0);
  vector<int> sizes(graph.numberOfVertices(), 1);

  for (int i = 0; i < edges.size(); i++) {
    Edge currentEdge = edges[i];
    cout<<"Examining edge "<<i<<" from "<<currentEdge.source<<" to "<<currentEdge.destination<<" weighted "<<currentEdge.weight<<endl;
    int root1 = segmentation.find(currentEdge.source);
    int root2 = segmentation.find(currentEdge.destination);
    float mInt = min(internalDifferences[root1] + ((float)k)/sizes[root1],
		     internalDifferences[root2] + ((float)k)/sizes[root2]);
    
    if (root1 != root2 && currentEdge.weight <= mInt) {
      cout<<"Fusing "<<root1<<" and "<<root2<<endl;
      int newRoot = segmentation.setUnion(root1,root2);
      cout<<"Fusion performed, new root "<<newRoot<<endl;
      internalDifferences[newRoot] = currentEdge.weight;
      sizes[newRoot] = sizes[root1] + sizes[root2];
    }
  }

  return segmentation;
}

WeightedGraph gridGraph(Mat_<uchar> &image) {
  WeightedGraph grid(image.cols*image.rows, 4);
  int colOffsets[] = {-1, 0, 1, 1};
  int rowOffsets[] = { 1, 1, 1, 0};

  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      int centerIndex = toRowMajor(image.cols, j,i);
      assert(centerIndex >= 0 && centerIndex < grid.numberOfVertices());
      int centerIntensity = image(i,j);
      
      for (int n = 0; n < 4; n++) {
	int neighborRow = i + rowOffsets[n];
	int neighborCol = j + colOffsets[n];
	
	if (neighborRow >= 0 && neighborRow < image.rows &&
	    neighborCol >= 0 && neighborCol < image.cols) {
	  int neighborIndex = toRowMajor(image.cols, neighborCol, neighborRow);
	  int neighborIntensity = image(neighborRow, neighborCol);
	  assert(neighborIndex >= 0 && neighborIndex < grid.numberOfVertices());
	  
	  grid.addEdge(centerIndex, neighborIndex, 
		       abs(centerIntensity - neighborIntensity));
	}
      }
    }
  }

  return grid;
}
