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

		line(imageToDrawOn, Point((int)floor(srcPos[1]), (int)floor(srcPos[0])), Point((int)floor(dstPos[1]), (int)floor(dstPos[0])), Scalar(0,0,0));
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
			line(imageToDrawOn, Point((int)floor(srcPos[1]), (int)floor(srcPos[0])), Point((int)floor(dstPos[1]), (int)floor(dstPos[0])), color);
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
		for (int j = 0; j < (int)graph.adjacencyLists[i].size(); j++) {
			os<<"("<<graph.adjacencyLists[i][j].destination<<","<<graph.adjacencyLists[i][j].weight<<")";
			if (j < (int)graph.adjacencyLists[i].size() - 1) {
				os<<", ";
			}
		}
		os<<"]"<<endl;
	}

	return os;
}

void connectedComponents(const WeightedGraph &graph, vector<int> &inConnectedComponent, int *nbCC) {
	*nbCC = 0;
	inConnectedComponent = vector<int>(graph.numberOfVertices(),-1);
	vector<bool> discovered(graph.numberOfVertices(), false);
	vector<int> stack;

	stack.reserve(graph.numberOfVertices());

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		if (!discovered[i]) {
			discovered[i] = true;

			inConnectedComponent[i] = *nbCC;

			stack.push_back(i);

			while (!stack.empty()) {
				int t = stack.back();
				stack.pop_back();

				for (int j = 0; j < (int)graph.getAdjacencyList(t).size(); j++) {
					HalfEdge edge = graph.getAdjacencyList(t)[j];

					if (!discovered[edge.destination]) {
						discovered[edge.destination] = true;

						inConnectedComponent[edge.destination] = *nbCC;

						stack.push_back(edge.destination);
					}
				}
			}
			(*nbCC)++;
		}
	}
}

bool connected(const WeightedGraph& graph) {
	int nbCC;

	connectedComponents(graph, vector<int>(), &nbCC);

	return nbCC == 1;
}

bool noLoops(const WeightedGraph& graph) {
	for (int i = 0; i < (int)graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		if (edge.source == edge.destination) {
			return false;
		}
	}

	return true;
}

bool bidirectional(const WeightedGraph& graph) {
	for (int i = 0; i < graph.numberOfVertices(); i++) {
		for (int j = 0; j < (int)graph.getAdjacencyList(i).size(); j++) {
			HalfEdge edge = graph.getAdjacencyList(i)[j];

			bool inOtherDir = false;

			for (int k = 0; k < (int)graph.getAdjacencyList(edge.destination).size(); k++) {
				HalfEdge other = graph.getAdjacencyList(edge.destination)[k];

				if (other.destination == i && abs(other.weight - edge.weight) <= 10E-8) {
					if (inOtherDir) {
						return false;
					} else {
						inOtherDir = true;
					}
				}
			}

			if (!inOtherDir) {
				return false;
			}
		}
	}

	return true;
}

void inducedSubgraphs(const WeightedGraph &graph, const vector<int> &inSubgraph, int numberOfSubgraphs, vector<int> &vertexIdx, vector<WeightedGraph> &subgraphs) {
	vector<int> subgraphSizes(numberOfSubgraphs,0);

	vertexIdx = vector<int>(graph.numberOfVertices(), -1);

	/*cout<<"vertexIdx: "<<vertexIdx.size()<<", subgraphSizes: "<<subgraphSizes.size()<<", inSubgraph: "<<inSubgraph.size()<<endl;

	cout<<"computing subgraph sizes and vertices indexes"<<endl;*/
	for (int i = 0; i < graph.numberOfVertices(); i++) {
		//cout<<"inSubgraph["<<i<<"] = "<<inSubgraph[i]<<endl;
		vertexIdx[i] = subgraphSizes[inSubgraph[i]];
		subgraphSizes[inSubgraph[i]]++;
	}

	subgraphs = vector<WeightedGraph>(numberOfSubgraphs);

	for (int i = 0; i < numberOfSubgraphs; i++) {
		subgraphs[i] = WeightedGraph(subgraphSizes[i]);
	}
	
	for (int i = 0; i < (int)graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		if (inSubgraph[edge.source] == inSubgraph[edge.destination]) {
			subgraphs[inSubgraph[edge.source]].addEdge(vertexIdx[edge.source], vertexIdx[edge.destination], edge.weight);
		}
	}
}

void WeightedGraph::copyEdges(const WeightedGraph &graph) {
	assert(graph.numberOfVertices() <= this->numberOfVertices());

	for (int i = 0; i < graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		this->addEdge(edge.source, edge.destination, edge.weight);
	}
}
