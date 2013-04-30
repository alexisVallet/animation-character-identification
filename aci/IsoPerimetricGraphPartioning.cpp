#include "IsoperimetricGraphPartitioning.h"

#define IGP_DEBUG false

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
static pair<int,double> ratioCutThreshold(const WeightedGraph &graph, int v0, Eigen::SparseMatrix<double> &L0, Eigen::VectorXd &d0, Eigen::VectorXd &degrees, Eigen::VectorXd &x0, Mat_<int> &s, double graphVolume) {
	// the initial best cut contains just v0, which we arbitrarily index at the last position
	vector<bool> isInSegment(x0.rows(), false);
	double previousBoundary = degrees(v0);
	double previousVolume = degrees(v0);

	for (int i = 0; i < x0.rows(); i++) {
		double internalWeights = 0;

		for (int j = 0; j < graph.getAdjacencyList(vertexMap(v0, s(i,0))).size(); j++) {
			HalfEdge edge = graph.getAdjacencyList(vertexMap(v0, s(i,0)))[j];
			
			if (edge.destination == v0) {
				internalWeights += edge.weight;
			} else if (isInSegment[reverseVertexMap(v0, edge.destination)]) {
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

	// the case where all vertices are in the segment, which should never happen because
	// of the volume constraint.
	cout<<"Error: all vertices are in the segment, this shouldn't happen"<<endl;
	exit(EXIT_FAILURE);
}

DisjointSetForest subgraphsIGP(const WeightedGraph &graph, double stop, vector<int> &inSubgraph, int numberOfSubgraphs, int maxRecursions) {
	vector<WeightedGraph> subgraphs;
	vector<int> vertexIdx;
	inducedSubgraphs(graph, inSubgraph, numberOfSubgraphs, vertexIdx, subgraphs);
	
	/*for (int i = 0; i < subgraphs.size(); i++) {
		cout<<"induced"<<i<<":"<<endl<<subgraphs[i]<<endl;
	}*/

	vector<DisjointSetForest> partitions(numberOfSubgraphs);

	for (int i = 0; i < numberOfSubgraphs; i++) {
		//cout<<"calling ipg on induced"<<i<<endl;
		partitions[i] = isoperimetricGraphPartitioning(subgraphs[i], stop, maxRecursions);
	}

	DisjointSetForest partition(graph.numberOfVertices());

	fusePartitions(graph, inSubgraph, vertexIdx, partitions, partition);

	return partition;
}

/**
 * Computes the isoperimetric graph partitioning of each connected component
 * in the graph, then fuses the results in a single partition.
 *
 * @param graph a possibly unconnected graph to partition
 * @param stop stopping parameter.
 * @return a partition of the graph.
 */
DisjointSetForest unconnectedIGP(const WeightedGraph &graph, double stop, int maxRecursions) {
	vector<WeightedGraph> components;
	vector<int> inConnectedComponent;
	int numberOfComponents;

	connectedComponents(graph, inConnectedComponent, &numberOfComponents);

	/*cout<<"g:"<<endl<<graph<<endl;
	cout<<"detected "<<numberOfComponents<<" connected components"<<endl;*/

	return subgraphsIGP(graph, stop, inConnectedComponent, numberOfComponents, maxRecursions);
}

DisjointSetForest isoperimetricGraphPartitioning(const WeightedGraph &graph, double stop, int maxRecursions) {
	if (maxRecursions == 0) {
		DisjointSetForest entireSegment(graph.numberOfVertices());

		for (int i = 1; i < graph.numberOfVertices(); i++) {
			entireSegment.setUnion(0, i);
		}

		return entireSegment;
	}
	// If G has no vertices we cannot partition it.
	if (graph.numberOfVertices() == 0) {
		cout<<"Cannot partition an empty graph."<<endl;
		exit(EXIT_FAILURE);
	}
	// if G has only one vertex, then it's in it own segment
	if (graph.numberOfVertices() == 1) {
		if (IGP_DEBUG) cout<<"trivial graph"<<endl;
		DisjointSetForest trivial(1);

		return trivial;
	}

	// We compute the laplacian of the graph and its ground vertex
	Eigen::VectorXd d;
	Eigen::SparseMatrix<double> L = sparseLaplacian(graph, true, d);
	if (IGP_DEBUG) {
		cout<<"L:"<<endl<<L<<endl;
		cout<<"d:"<<endl<<d<<endl;
	}
	int v0;
	d.maxCoeff(&v0);
	if (IGP_DEBUG) cout<<"v0 = "<<v0<<endl;

	// We remove the line and column of v0 in L to obtain L0.
	Eigen::SparseMatrix<double> L0;

	removeLineCol(L, v0, L0);

	if (IGP_DEBUG) cout<<"L0:"<<endl<<L0<<endl;

	// We remove the degree of v0 from g to obtain d0
	Eigen::VectorXd d0(d.size() - 1);
	
	d0.head(v0) = d.head(v0);
	d0.tail(d0.size() - v0) = d.tail(d0.size() - v0);

	if (IGP_DEBUG) cout<<"d0:"<<endl<<d0<<endl;

	// We now solve the linear system L0 * x0 = d0 for x0
	assert(symmetric(L0));
	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;

	solver.compute(L0);

	Eigen::VectorXd x0 = solver.solve(d0);

	if (IGP_DEBUG) cout<<"x0:"<<endl<<x0<<endl;

	// We threshold x0 to obtain the cut with best isoperimetric ratio
	// First we need to sort x0 and compute the volume of the graph.
	double graphVolume = d.sum();
	Mat_<double> cvx0(x0.size(), 1);
	Mat_<int> s;

	for (int i = 0; i < x0.size(); i++) {
		cvx0(i,0) = x0(i);
	}

	sortIdx(cvx0, s, CV_SORT_ASCENDING + CV_SORT_EVERY_COLUMN);

	pair<int,double> cut = ratioCutThreshold(graph, v0, L0, d0, d, x0, s, graphVolume);

	//cout<<"Best cut at "<<cut.first<<" with ratio "<<cut.second<<endl;

	if (cut.second < stop) {
		if (IGP_DEBUG) cout<<"good enough cut at "<<cut.second<<endl;
		//cout<<"cut good enough, returning bipartition"<<endl;
		DisjointSetForest bipartition(graph.numberOfVertices());
		
		// fuse all the elements of the first segment with the ground
		for (int i = 0; i <= cut.first; i++) {
			bipartition.setUnion(v0, vertexMap(v0, s(i,0)));
		}

		// fuse all the others with each other
		for (int i = cut.first + 2; i < x0.size(); i++) {
			bipartition.setUnion(vertexMap(v0, s(cut.first + 1)), vertexMap(v0, s(i,0)));
		}

		return bipartition;
	}

	vector<int> inSegment(graph.numberOfVertices(),-1);

	for (int i = 0; i <= cut.first; i++) {
		inSegment[vertexMap(v0, s(i,0))] = 0;
	}

	for (int i = cut.first + 1; i < x0.size(); i++) {
		inSegment[vertexMap(v0, s(i,0))] = 1;
	}

	inSegment[v0] = 0;

	//cout<<"bipartition not good enough, computing subgraphs"<<endl;

	vector<int> vertexIdx;
	vector<WeightedGraph> subgraphs;

	//cout<<"computing induced subgraphs"<<endl;
	inducedSubgraphs(graph, inSegment, 2, vertexIdx, subgraphs);

	//cout<<"g1:"<<endl<<subgraphs[0]<<endl;
	//cout<<"g2:"<<endl<<subgraphs[1]<<endl;

	vector<DisjointSetForest> partitions(2);

	//cout<<"computing connected components and recursive calls"<<endl;
	for (int i = 0; i < 2; i++) {
		partitions[i] = unconnectedIGP(subgraphs[i], stop, maxRecursions - 1);
	}

	DisjointSetForest partition(graph.numberOfVertices());

	//cout<<"fusing partitions from recursive calls"<<endl;
	fusePartitions(graph, inSegment, vertexIdx, partitions, partition);

	return partition;
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