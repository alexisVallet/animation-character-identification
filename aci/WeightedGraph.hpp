/**
 * Weighted graph data structure represented as adjacency
 * lists. Lists are stored as vectors for efficiency, as our
 * graphs in the use case of image segmentation have a limited
 * degree.
 */
#ifndef _WEIGHTEDGRAPH_HPP_
#define _WEIGHTEDGRAPH_HPP_

#include <cmath>
#include <assert.h>
#include <iostream>
#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>
#include <boost/graph/adjacency_list.hpp>

using namespace std;
using namespace cv;
using namespace boost;

struct Edge {
  int source;
  int destination;
  float weight;
};

struct HalfEdge {
  int destination;
  float weight;
};

class WeightedGraph {
private:
  vector<vector<HalfEdge> > adjacencyLists;
  vector<Edge> edges;

public:
  WeightedGraph(); // should not be called
  /**
   * Initializes the graph with a given number of vertices and
   * an optional upper bound on the degree of vertices of the
   * graph.
   *
   * @param numberOfVertices the number of vertices of the graph
   * @param maxDegree an upper bound on the degree of vertices in
   * the graph.
   */
  WeightedGraph(int numberOfVertices, int maxDegree = -1);
  /**
   * Adds an edge to the graph. In the case of an undirected graph,
   * the order of source and destination does not matter.
   *
   * @param source the source vertex of the edge.
   * @param destination the destination vertex of the edge.
   * @param weight the weight of the edge.
   */
  void addEdge(int source, int destination, float weight);
  /**
   * Computes and returns a vector containing all the edges in the
   * graph.
   *
   * @return a vector containing all the edges in the graph.
   */
  const vector<Edge> &getEdges() const;
  /**
   * The number of vertices of the graph.
   *
   * @return the number of vertices of the graph.
   */
  int numberOfVertices() const;
  /**
   * Draws the graph on an image.
   *
   * @param verticesPositions vertices positions on the image
   * @param imageToDrawOn image the graph will be drawn over
   */
  void drawGraph(vector<Vec<float,2> > verticesPositions, Mat &imageToDrawOn);
  /**
   * Converts this graph into a boost adjacency list. Drops both
   * weights and vertices labels.
   *
   * @return adjacency list representing the same grpah as this.
   */
  adjacency_list<vecS,vecS,bidirectionalS,property<vertex_index_t, int> > toBoostGraph();
};

#endif
