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

void WeightedGraph::addEdge(int source, int destination, float weight = 1) {
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
  for (int i = 0; i < (int)this->edges.size(); i++) {
    Edge edge = this->edges[i];
    Vec<float,2> srcPos = verticesPositions[edge.source];
    Vec<float,2> dstPos = verticesPositions[edge.destination];

    line(imageToDrawOn, Point(floor(srcPos[1]), floor(srcPos[0])), Point(floor(dstPos[1]), floor(dstPos[0])), Scalar(0,0,0));
  }
}

void WeightedGraph::drawGraphWithEmbedding(vector<Vec<float,2> > verticesPositions, Mat &imageToDrawOn, graph_t boostGraph, embedding_t embedding) {
	for (int i = 0; i < this->numberOfVertices(); i++) {
		property_traits<embedding_t>::value_type::const_iterator it;
		Scalar color = Scalar(rand()%255,rand()%255,rand()%255);

		for (it = embedding[i].begin(); it != embedding[i].end(); it++) {
			int dst =
				get(vertex_index, boostGraph, target(*it, boostGraph)) ==  i ? 
				get(vertex_index, boostGraph, source(*it, boostGraph)) :
				get(vertex_index, boostGraph, target(*it, boostGraph));
		    Vec<float,2> srcPos = verticesPositions[i];
			Vec<float,2> dstPos = verticesPositions[dst];

			cout<<"Drawing edge between "<<i<<" and "<<dst<<endl;
			line(imageToDrawOn, Point(floor(srcPos[1]), floor(srcPos[0])), Point(floor(dstPos[1]), floor(dstPos[0])), color);
			imshow("graph", imageToDrawOn);
			waitKey(0);
		}
	}
}

graph_t WeightedGraph::toBoostGraph() {
  graph_t
    graph(this->adjacencyLists.size());
  
  for (int i = 0; i < (int)this->edges.size(); i++) {
    add_edge(this->edges[i].source, this->edges[i].destination, graph);
  }

  return graph;
}

const vector<HalfEdge> &WeightedGraph::getAdjacencyList(int vertex) const {
	return this->adjacencyLists[vertex];
}

ostream &operator<<(ostream &os, const WeightedGraph &graph) {
	for (int i = 0; i < graph.numberOfVertices(); i++) {
		os<<i<<" : [";
		for (int j = 0; j < graph.adjacencyLists[i].size(); j++) {
			os<<graph.adjacencyLists[i][j].destination;
			if (j < graph.adjacencyLists[i].size() - 1) {
				os<<", ";
			}
		}
		os<<"]"<<endl;
	}

	return os;
}