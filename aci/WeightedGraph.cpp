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
