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
DisjointSetForest isoperimetricGraphPartitioning(const WeightedGraph &graph, double stop) {
	// Compute laplacian and degrees
	cout<<"computing laplacian"<<endl;
	Eigen::VectorXd degrees(graph.numberOfVertices());
	Eigen::SparseMatrix<double> matrix = sparseLaplacian(graph, true, degrees);
	
	// determine ground vertex as the maximum degree vertex
	int ground;

	degrees.maxCoeff(&ground);

	cout<<"ground vertex is "<<ground<<endl;
	// remove the ground vertex line/column from the laplacian
	// and its degree so the eigenvalue problem becomes a simple linear
	// system
	cout<<"removing ground from laplacian, degrees"<<endl;
	typedef Eigen::Triplet<double> T;
	vector<T> tripletList;

	tripletList.reserve(matrix.nonZeros());
	
	for (int k = 0; k < matrix.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(matrix,k); it; ++it) {
			if (it.row() != ground && it.col() != ground) {
				int newRow = it.row() > ground ? it.row() - 1 : it.row();
				int newCol = it.col() > ground ? it.col() - 1 : it.col();

				tripletList.push_back(T(newRow, newCol, it.value()));
			}
		}
	}

	Eigen::SparseMatrix<double> L0(graph.numberOfVertices() - 1, graph.numberOfVertices() - 1);

	L0.setFromTriplets(tripletList.begin(), tripletList.end());

	Eigen::VectorXd d0(graph.numberOfVertices() - 1, 1);

	d0.head(ground) = degrees.head(ground);
	int tailLength = d0.size() - ground;
	d0.tail(tailLength) = degrees.tail(tailLength);

	// solve the linear system using the conjugate gradient method
	cout<<"solving linear system using conjugate gradient"<<endl;
	cout<<"checking positive definiteness"<<endl;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > chol;

	chol.analyzePattern(L0);
	chol.factorize(L0);

	assert(chol.info() == Eigen::Success);
	cout<<" the matrix is positive definite"<<endl;

	int maxIterations = 10000;
	Eigen::ConjugateGradient<Eigen::SparseMatrix<double> > cg;

	cg.compute(L0);
	Eigen::VectorXd x0(graph.numberOfVertices() - 1);
	x0 = cg.solve(d0);

	cout<<"thresholding to find the bipartition"<<endl;
	// thresholding to find the best ratio-cut
	vector<double> sortedIdx(graph.numberOfVertices() - 1);

	for (int i = 0; i < graph.numberOfVertices() - 1; i++) {
		sortedIdx[i] = x0(i);
	}

	sort(sortedIdx.begin(), sortedIdx.end());

	int bestCut = 0;
	vector<bool> currentSegment(graph.numberOfVertices() - 1, false);
	currentSegment[0] = true;
	double currentCutSize = d0(sortedIdx[0],0);
	double currentSegmentVol = currentCutSize;
	double bestCutRatio = 1;

	for (int i = 1; i < graph.numberOfVertices() - 1; i++) {
		currentSegment[i] = true;
		int vertex = sortedIdx[i];
		// the new segment volume simply takes the degree of the new vertex
		currentSegmentVol += d0(vertex);
		// the new cut size however require a bit more computation. Provided
		// the number of edges is a constant factor away from the number of vertices,
		// (as for planar graphs for instance) this is still linear time for this loop
		// in the number of vertices.
		currentCutSize += d0(vertex);
		int actualVertex = vertex < ground ? vertex : vertex + 1; // while the x0 vector has the ground removed, the graph doesn't
		// for each neighboring vertex that's in the current segment, subtract twice the
		// weight of the edge to the cut size, once for the degree of the source and once for the degree
		// of the destination.
		for (int j = 0; j < (int)graph.getAdjacencyList(actualVertex).size(); j++) {
			HalfEdge edge = graph.getAdjacencyList(actualVertex)[j];
			int neighbor = j < ground ? j : j - 1; // once again keeping track of the ground

			if (currentSegment[neighbor]) {
				currentCutSize -= 2 * edge.weight;
			}
		}
		double newCutRatio = currentCutSize / currentSegmentVol;

		if (newCutRatio < bestCutRatio) {
			bestCutRatio = newCutRatio;
			bestCut = i;
		}
	}

	cout<<"best cut "<<bestCut<<endl;

	// if the cut ratio is lower than the stopping parameter, stop the recursion and
	// compute the disjoint set forest corresponding to the bipartition. The ground
	// belongs to the previously computed segment.
	if (bestCutRatio < stop) {
		DisjointSetForest bipartition(graph.numberOfVertices());

		// fuse every vertex in the segment with the ground vertex
		for (int i = 0; i <= bestCut; i++) {
			int vertex = sortedIdx[i] < ground ? sortedIdx[i] : sortedIdx[i] + 1;

			bipartition.setUnion(ground, vertex);
		}
		int firstOut = -1;
		// fuse every other one with each other to form the complement
		for (int i = bestCut + 1; i < graph.numberOfVertices() - 1; i++) {
			int vertex = sortedIdx[i] < ground ? sortedIdx[i] : sortedIdx[i] + 1;

			if (firstOut < 0) {
				firstOut = vertex;
			} else {
				bipartition.setUnion(firstOut, vertex);
			}
		}

		return bipartition;
	}

	// If the stopping criterion hasn't been reached, call the procedure recursively on
	// the subgraphs induced by S and its complement respectively. First, we need to
	// acutally compute S and its complement. S is basically sortedIdx up to the best cut,
	// with the addition of the ground vertex. The complement is the rest.
	vector<int> s1(sortedIdx.begin(), sortedIdx.begin() + bestCut + 1);
	s1.push_back(ground);
	vector<int> s2(sortedIdx.begin() + bestCut + 1, sortedIdx.end());
	// We then compute the mapping which associate the value of a vertex in G to its index
	// in either graph. It's essentially the inverse of the concatenation of s1 and s2.

	vector<int> s1ps2(s1.size() + s2.size());

	for (int i = 0; i < (int)s1.size(); i++) {
		s1ps2[i] = s1[i];
	}
	for (int i = 0; i < (int)s2.size(); i++) {
		s1ps2[s1.size() + i] = s2[i];
	}

	// This is basically a dirty trick to emulate a sum type.
	enum Segment { S1, S2 }; // ;_; why am I not programming in Haskell
	vector<pair<Segment,int> > f(s1.size() + s2.size());

	for (int i = 0; i < (int)(s1.size() + s2.size()); i++) {
		if (i < (int)s1.size()) {
			f[s1ps2[i]] = pair<Segment,int>(S1, i);
		} else {
			f[s1ps2[i]] = pair<Segment,int>(S2, i - s1.size());
		}
	}
	// now f associates to each vertex v in G either:
	// - its index in s1 tagged by S1 so I actually know it's in s1
	// - its index in s2 tagged by S2 otherwise
	// from these mappings we can now actually construct g1 and g2 by enumerating
	// the edges in G.
	WeightedGraph
		g1(s1.size()),
		g2(s2.size());

	for (int i = 0; i < (int)graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];
		pair<Segment,int> srcSeg = f[edge.source];
		pair<Segment,int> dstSeg = f[edge.destination];

		// If the source and destination are not in the same segment, we ignore them.
		// If they are in the same, we add them to the corresponding graph with the
		// same weight.
		if (srcSeg.first == dstSeg.first) {
			if (srcSeg.first == S1) {
				g1.addEdge(srcSeg.second, dstSeg.second, edge.weight);
			} else {
				g2.addEdge(srcSeg.second, dstSeg.second, edge.weight);
			}
		}
	}

	// We can now finally call the procedure recursively.
	DisjointSetForest p1 = isoperimetricGraphPartitioning(g1, stop);
	DisjointSetForest p2 = isoperimetricGraphPartitioning(g2, stop);
	// We combine the two segmentations and return it
	assert(p1.getNumberOfElements() + p2.getNumberOfElements() == graph.numberOfVertices());
	DisjointSetForest segmentation(graph.numberOfVertices());
	// the indexes of vertices in G from those in g1 and g2 are mapped by s1 and s2
	// respectively.
	for (int i = 0; i < (int)graph.getEdges().size(); i++) {
		Edge edge = graph.getEdges()[i];
		pair<Segment,int> srcSeg = f[edge.source];
		pair<Segment,int> dstSeg = f[edge.destination];

		// If the source and destination are in the same segment, fuse them iff they
		// are in the same subsegment in p1 or p2.
		if (srcSeg.first == dstSeg.first) {
			if (srcSeg.first == S1) {
				if (p1.find(srcSeg.second) == p1.find(dstSeg.second)) {
					segmentation.setUnion(edge.source, edge.destination);
				}
			} else {
				if (p2.find(srcSeg.second) == p2.find(dstSeg.second)) {
					segmentation.setUnion(edge.source, edge.destination);
				}
			}
		}
	}

	return segmentation;
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