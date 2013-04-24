/**
 * Weighted graph data structure represented as adjacency
 * lists. Lists are stored as vectors for efficiency, as our
 * graphs in the use case of image segmentation have a limited
 * degree. Also keeps an edge list data structure for efficiently
 * listing edges in the graph.
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
#include <boost/graph/properties.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

using namespace std;
using namespace cv;
using namespace boost;

// convenience typedefs for boost graph datatypes
typedef adjacency_list<vecS,vecS,bidirectionalS,property<vertex_index_t, int> > graph_t;

typedef vector<vector<graph_traits<adjacency_list<vecS,vecS,bidirectionalS,property<vertex_index_t, int> > >::edge_descriptor> > embedding_storage_t;

typedef iterator_property_map<vector<vector<graph_traits<adjacency_list<vecS,vecS,bidirectionalS,property<vertex_index_t, int> > >::edge_descriptor> >::iterator,property_map<adjacency_list<vecS,vecS,bidirectionalS,property<vertex_index_t, int> >,vertex_index_t>::type> embedding_t;

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
   * Draws the graph on an image, vertex by vertex adding each edge in its circular
   * ordering as defined by a planar embedding. Displays the intermediary image at each
   * step, waiting for user input between each step. This method should therefore only
   * be used for debugging purposes.
   *
   * @param verticesPositions vertices positions on the image
   * @param imageToDrawOn image the graph will be drawn over
   * @param embedding a planar embedding of the graph, as returned by 
   * boost::graph::boyer_myrvold_planarity_test for instance.
   */
  void drawGraphWithEmbedding(vector<Vec<float,2> > verticesPositions, Mat &imageToDrawOn, graph_t boostGraph, embedding_t embedding);
  /**
   * Converts this graph into a boost adjacency list. Drops both
   * weights and vertices labels.
   *
   * @return adjacency list representing the same grpah as this.
   */
  graph_t toBoostGraph();

  /**
   * Returns the adjacency list of a specific vertex. In the case of
   * unidirectional representation, this does not return all adjacent
   * vertices. Use a bidirectional representation for this.
   */
  const vector<HalfEdge> &getAdjacencyList(int vertex) const;

  friend ostream &operator<<(ostream &os, const WeightedGraph &graph);
};

/**
 * Computes the connected components of a graph using a simple DFS procedure.
 *
 * @param graph the graph to compute connected components from.
 * @param inConnectedComponent output vector which associates to each vertex the
 * index of the connected component if belongs to.
 * @param vertexIdx output map which associates to each vertex in the graph its
 * index in the associated subgraph.
 */
void connectedComponents(const WeightedGraph &graph, vector<int> &inConnectedComponent, int *nbCC);

/**
 * Checks that a graph is connected.
 */
bool connected(const WeightedGraph& graph);

/**
 * Checks that a graph contains no loops.
 */
bool noLoops(const WeightedGraph& graph);

/**
 * Checks that a graph has a bidirectional representation.
 */
bool bidirectional(const WeightedGraph& graph);

#endif
