#include "WeightedGraph.hpp"

WeightedGraph::WeightedGraph() {
  
}

WeightedGraph::WeightedGraph(int numberOfVertices, int maxDegree) 
  : adjacencyLists(numberOfVertices)
{
  if (maxDegree > 0) {
    for (int i = 0; i < numberOfVertices; i++) {
      this->adjacencyLists[i].reserve(maxDegree);
    }
    this->edges.reserve(numberOfVertices*maxDegree);
  }
}

void WeightedGraph::addEdge(int source, int destination, float weight) {
  HalfEdge toAdd;

  assert(source >= 0 && source < this->numberOfVertices());
  assert(destination >= 0 && destination < this->numberOfVertices());

  toAdd.destination = destination;
  toAdd.weight = weight;
  this->adjacencyLists[source].push_back(toAdd);

  Edge fullEdge;

  fullEdge.source = source;
  fullEdge.destination = destination;
  fullEdge.weight = weight;
  this->edges.push_back(fullEdge);
}

const vector<Edge> &WeightedGraph::getEdges() const {
  return this->edges;
}

int WeightedGraph::numberOfVertices() const {
  return this->adjacencyLists.size();
}

void WeightedGraph::drawGraph(vector<Vec<float,2> > verticesPositions, Mat &imageToDrawOn) {
  for (int i = 0; i < this->edges.size(); i++) {
    Edge edge = this->edges[i];
    Vec<float,2> srcPos = verticesPositions[edge.source];
    Vec<float,2> dstPos = verticesPositions[edge.destination];

    line(imageToDrawOn, Point(srcPos[1], srcPos[0]), Point(dstPos[1], dstPos[0]), Scalar(0,0,0));
  }
}

adjacency_list<vecS,vecS,bidirectionalS,property<vertex_index_t, int> > WeightedGraph::toBoostGraph() {
  adjacency_list<vecS,vecS,bidirectionalS,property<vertex_index_t, int> > 
    graph(this->adjacencyLists.size());
  
  for (int i = 0; i < this->edges.size(); i++) {
    add_edge(this->edges[i].source, this->edges[i].destination, graph);
  }

  return graph;
}
