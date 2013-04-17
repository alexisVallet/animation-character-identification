#include "IsoperimetricGraphPartitioning.h"

WeightedGraph removeIsolatedVertices(WeightedGraph &graph, vector<int> &vertexMap) {
	// first we count the number of non-isolated vertices in the graph, filling
	// vertexMap appropriately.
	vertexMap = vector<int>(graph.numberOfVertices());
	int nonIsolated = 0;

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		if (!graph.getAdjacencyList(i).empty()) {
			vertexMap[i] = nonIsolated;
			nonIsolated++;
		} else {
			vertexMap[i] = -1;
		}
	}

	WeightedGraph connected(nonIsolated);

	for (int i = 0; i < (int)graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];
		connected.addEdge(vertexMap[edge.source], vertexMap[edge.destination], edge.weight);
	}

	return connected;
}

// Remove a line and column with the same index in a sparse matrix.
static void removeLineCol(const Eigen::SparseMatrix<double> &L, int v0, Eigen::SparseMatrix<double> &L0) {
	typedef Eigen::Triplet<double> T;
	vector<T> tripletList;
	tripletList.reserve(L.nonZeros());

	for (int k = 0; k < L.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(L,k); it; ++it) {
			if (it.row() != v0 && it.col() != v0) {
				int newRow = it.row() < v0 ? it.row() : it.row() - 1;
				int newCol = it.col() < v0 ? it.col() : it.col() - 1;

				tripletList.push_back(T(newRow, newCol, it.value()));
			}
		}
	}

	L0 = Eigen::SparseMatrix<double>(L.rows() - 1, L.cols() - 1);

	L0.setFromTriplets(tripletList.begin(), tripletList.end());
}

/**
 * maps vertex index v from the space where the ground vertex v0 has been
 * removed to the space where the ground vertex is still there.
 *
 * @param v0 the ground vertex
 * @param v the vertex to map
 * @return the corresponding vertex in the larger graph
 */
static int vertexMap(int v0, int v) {
	return v < v0 ? v : v + 1;
}

/**
 * Reverse operation to vertexMap. Maps a vertex in the larger graph to
 * its associated vertex in the graph with v0 removed.
 *
 * @param v0 the ground vertex
 * @param v the vertex to map
 * @param the corresponding vertex in the smaller graph
 */
static int reverseVertexMap(int v0, int v) {
	assert(v0 != v);

	return v < v0 ? v : v - 1;
}

/**
 * computed the ratio cut threshold of solution vector x0, given the laplacian matrix with ground v0 removed
 * L0, the degrees vector with ground removed d0, and a sorting of the vertices by their value in x0 s.
 * The best ratio cut is the cut S with smallest isoperimetric ratio such that Vol(S) <= Vol(V)/2.
 *
 * @param graph the graph being segmented
 * @param v0 ground vertex
 * @oaram L0 laplacian matrix of the graph with ground vertex removed
 * @param d0 degrees vector of the graph with ground vertex removed
 * @param x0 the solution to L0 * x0 = d0
 * @param s indexes of elements of x0 sorted by ascending value of their value in x0.
 * @param graphVolume the total graph volume
 * @return the best cut index in the sorted x0, along with its isoperimetric ratio.
 */
static pair<int,double> ratioCutThreshold(const WeightedGraph &graph, int v0, Eigen::SparseMatrix<double> &L0, Eigen::VectorXd &d0, Eigen::VectorXd &x0, Mat_<int> &s, double graphVolume) {
	// the initial best cut contains just the first vertex in the sorted x0
	vector<bool> isInSegment(x0.rows(), false);
	isInSegment[0] = true;
	double previousBoundary = d0(s(0,0));
	double previousVolume = d0(s(0,0));

	for (int i = 1; i < x0.rows(); i++) {
		double internalWeights = 0;

		for (int j = 0; j < graph.getAdjacencyList(vertexMap(v0, s(i,0))).size(); j++) {
			HalfEdge edge = graph.getAdjacencyList(vertexMap(v0, s(i,0)))[j];
			
			if (edge.destination != v0 && isInSegment[reverseVertexMap(v0, edge.destination)]) {
				internalWeights += edge.weight;
			}
		}

		// because the weights are non negative, it is clear that the isoperimetric
		// ratio is decreasing at each iteration.
		double newBoundary = previousBoundary + d0(s(i,0)) - 2 * internalWeights;
		double newVolume = previousVolume + d0(s(i,0));

		// If the currentVolume is greater than half the total volume of the graph, we stop
		// and return the previous best cut.
		if (newVolume > graphVolume/2) {
			return pair<int,double>(i-1,previousBoundary/previousVolume);
		}

		// otherwise we update the volume and boundary
		previousBoundary = newBoundary;
		previousVolume = newVolume;
	}

	//the special case where the best cut is all the vertices but the ground
	return pair<int,double>(x0.rows() - 1, previousBoundary/previousVolume);
}

/**
 * Computes the connected components of a graph using a simple DFS procedure.
 *
 * @param graph the graph to compute connected components from.
 * @param inConnectedComponent output vector which associates to each vertex the
 * index of the connected component if belongs to.
 * @param vertexIdx output map which associates to each vertex in the graph its
 * index in the associated subgraph.
 */
void connectedComponents(const WeightedGraph &graph, vector<int> &inConnectedComponent, vector<int> &vertexIdx) {
	int nbCC = 0;
	inConnectedComponent = vector<int>(graph.numberOfVertices(),-1);
	vector<bool> discovered(graph.numberOfVertices(), false);
	vector<int> stack;

	stack.reserve(graph.numberOfVertices());

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		if (!discovered[i]) {
			discovered[i] = true;
			inConnectedComponent[i] = nbCC;

			stack.push_back(i);

			while (!stack.empty()) {
				int t = stack.back();
				stack.pop_back();

				for (int j = 0; j < graph.getAdjacencyList(t).size(); j++) {
					HalfEdge edge = graph.getAdjacencyList(t)[j];

					if (!discovered[edge.destination]) {
						discovered[edge.destination] = true;
						inConnectedComponent[edge.destination] = nbCC;
						stack.push_back(edge.destination);
					}
				}
			}

			nbCC++;
		}
	}
}

/** 
 * Computes the subgraphs induced by a specific partition.
 *
 * @param graph the graph to compute the subgraphs from
 * @param inSubgraph a graph.numberOfVertices() sized vector which associates to each vertex in the larger
 * graph the index of the subgraph it belongs to.
 * @param vertexIdx output vector containing a mapping from vertices in the graph to vertices in the corresponding
 * subgraph.
 * @param subgraphs output graphs which will be populated with the subgraphs.
 */ 
void inducedSubgraphs(const WeightedGraph &graph, const vector<int> &inSubgraph, int numberOfSubgraphs, vector<int> &vertexIdx, vector<WeightedGraph> &subgraphs) {
	vector<int> subgraphSizes(numberOfSubgraphs,0);
	vertexIdx = vector<int>(inSubgraph.size(), -1);

	for (int i = 0; i < inSubgraph.size(); i++) {
		vertexIdx[i] = subgraphSizes[i];
		subgraphSizes[i]++;
	}

	subgraphs = vector<WeightedGraph>(numberOfSubgraphs);

	for (int i = 0; i < numberOfSubgraphs; i++) {
		subgraphs[i] = WeightedGraph(subgraphSizes[i]);
	}

	for (int i = 0; i < graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		if (inSubgraph[edge.source] == inSubgraph[edge.destination]) {
			subgraphs[inSubgraph[edge.source]].addEdge(vertexIdx[edge.source], vertexIdx[edge.destination], edge.weight);
		}
	}
}

DisjointSetForest subgraphsIGP(const WeightedGraph &graph, vector<int> &inSubgraph, int numberOfSubgraphs, vector<int> &vertexIdx) {
	vector<wWeightedGraph> subgraphs;
	inducedSubgraphs(graph, inSubgraph, numberOfSubgraphs, vertexIdx, subgraphs);
}

/**
 * Computes the isoperimetric graph partitioning of each connected component
 * in the graph, then fuses the results in a single partition.
 *
 * @param graph a possibly unconnected graph to partition
 * @param stop stopping parameter.
 * @return a partition of the graph.
 */
DisjointSetForest unconnectedIGP(const WeightedGraph &graph, double stop) {
	vector<WeightedGraph> components;
	vector<int> inConnectedComponent;
	vector<int> vertexIdx;

	connectedComponents(graph, components, inConnectedComponent, vertexIdx);

	vector<DisjointSetForest> partitions(components.size());

	for (int i = 0; i < partitions.size(); i++) {
		partitions[i] = isoperimetricGraphPartitioning(components[i], stop);
	}

	for (int i = 0; i < 
}

DisjointSetForest isoperimetricGraphPartitioning(const WeightedGraph &graph, double stop) {
	// If G has no vertices we cannot partition it.
	if (graph.numberOfVertices() == 0) {
		cout<<"Cannot partition an empty graph."<<endl;
		exit(EXIT_FAILURE);
	}
	// if G has only one vertex, then it's in it own segment
	if (graph.numberOfVertices() == 1) {
		DisjointSetForest trivial(1);

		return trivial;
	}

	// We compute the laplacian of the graph and its ground vertex
	Eigen::VectorXd d;
	Eigen::SparseMatrix<double> L = sparseLaplacian(graph, true, d);
	cout<<"L:"<<endl<<L<<endl;
	int v0;
	d.maxCoeff(&v0);
	cout<<"v0 = "<<v0<<endl;

	// We remove the line and column of v0 in L to obtain L0.
	Eigen::SparseMatrix<double> L0;

	removeLineCol(L, v0, L0);

	cout<<"L0:"<<endl<<L0<<endl;

	// We remove the degree of v0 from g to obtain d0
	Eigen::VectorXd d0(d.size() - 1);
	
	d0.head(v0) = d.head(v0);
	d0.tail(d0.size() - v0) = d.tail(d0.size() - v0);

	cout<<"d0:"<<endl<<d0<<endl;

	// We now solve the linear system L0 * x0 = d0 for x0
	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> cgSolver;
	Eigen::VectorXd x0 = cgSolver.compute(L0).solve(d0);

	cout<<"x0:"<<endl<<x0<<endl;

	// We threshold x0 to obtain the cut with best isoperimetric ratio
	// First we need to sort x0 and compute the volume of the graph.
	double graphVolume = d.sum();
	Mat_<double> cvx0(x0.size(), 1);
	Mat_<int> s;

	for (int i = 0; i < x0.size(); i++) {
		cvx0(i,0) = x0(i);
	}

	sortIdx(cvx0, s, CV_SORT_ASCENDING + CV_SORT_EVERY_COLUMN);

	cout<<"s:"<<endl<<s<<endl;

	pair<int,double> cut = ratioCutThreshold(graph, v0, L0, d0, x0, s, graphVolume);

	cout<<"Best cut at "<<cut.first<<" with ratio "<<cut.second<<endl;

	
}

DisjointSetForest addIsolatedVertices(WeightedGraph &graph, DisjointSetForest &segmentation, vector<int> &vertexMap) {
	assert(graph.numberOfVertices() == vertexMap.size());
	DisjointSetForest result(graph.numberOfVertices());

	// we first reproduce the segmentation in the larger forest, ignoring isolated vertices
	for (int i = 0; i < (int)graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];

		if (segmentation.find(vertexMap[edge.source]) == segmentation.find(vertexMap[edge.destination])) {
			result.setUnion(edge.source, edge.destination);
		}
	}

	int firstIsolated = -1;

	// We then fuse the isolated vertices in their own component
	for (int i = 0; i < (int)vertexMap.size(); i++) {
		if (vertexMap[i] < 0) {
			if (firstIsolated < 0) {
				firstIsolated = i;
			} else {
				result.setUnion(firstIsolated, i);
			}
		}
	}

	return result;
}

Mat_<double> conjugateGradient(SparseMat_<double> &A, Mat_<double> &b, Mat_<double> &x) {
	Mat_<double> r = b - sparseMul(A, x);
	Mat_<double> p = r.clone();
	double rsold = r.dot(r);

	for (int i = 0; i < 10E06; i++) {
		Mat_<double> Ap = sparseMul(A, p);
		double alpha = rsold / p.dot(Ap);
		x = x + alpha * p;
		r = r - alpha * Ap;
		double rsnew = r.dot(r);

		cout<<"iteration "<<i<<" rsnew ="<<rsnew<<endl;

		if (sqrt(rsnew) < 1E-10) {
			break;
		}

		p = r + (rsnew/rsold)*p;
		rsold = rsnew;
	}

	return x;
}