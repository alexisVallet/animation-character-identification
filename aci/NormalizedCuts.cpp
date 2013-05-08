#include "NormalizedCuts.h"

/**
 * Computes the cut to evector with the best normalized cut ratio. If the graph has a constant
 * bound on the unweighted degree of its vertices, this runs in O(n) time where n is the number
 * of vertices in the graph.
 *
 * @param graph the graph from which the eigenvector was computed.
 * @param degrees degrees of vertices in the graph.
 * @param evector eigenvector to compute normalized cut ratio from. Is assumed
 * to be sorted by sorting.
 * @param sorting specifies the original index of values before evector was sorted.
 * @return index of the best cut in the sorted vector, inclusive.
 */
static pair<int,double> normalizedCutThreshold(const WeightedGraph &graph, const VectorXd &degrees, const VectorXd &evector, const vector<int> sorting, vector<int> &inSubgraph) {
	// keeps track of cut(A,B), assoc(A,V), assoc(B,V)
	cout<<"initialization"<<endl;
	vector<int> isInSubgraph(graph.numberOfVertices(), 1);
	isInSubgraph[sorting[0]] = 0;
	double cutAB = degrees(sorting[0]);
	double assocAV = degrees(sorting[0]);
	double assocBV = degrees.sum() - degrees(sorting[0]);
	double bestRatio = cutAB/assocAV + cutAB/assocBV;
	int bestCut = 0;

	cout<<"computation"<<endl;
	for (int i = 1; i < graph.numberOfVertices(); i++) {
		assocAV += degrees(sorting[i]);
		assocBV -= degrees(sorting[i]);

		double internalWeights = 0;

		for (int j = 0; j < graph.getAdjacencyList(sorting[i]).size() - 1; j++) { // -1 because we don't want the entire graph
			HalfEdge edge = graph.getAdjacencyList(sorting[i])[j];

			if (isInSubgraph[edge.destination] == 0) {
				internalWeights += edge.weight;
			}
		}

		cutAB += degrees(sorting[i]) - 2 * internalWeights;

		double newRatio = cutAB/assocAV + cutAB/assocBV;

		if (newRatio < bestRatio) {
			bestRatio = newRatio;
			bestCut = i;
		}

		isInSubgraph[sorting[i]] = 0;
	}

	cout<<"computing inSubgraph vector"<<endl;

	inSubgraph = vector<int>(graph.numberOfVertices(), 1);

	for (int i = 0 ; i <= bestCut; i++) {
		inSubgraph[sorting[i]] = 0;
	}

	return pair<int,double>(bestCut,bestRatio);
}

static bool compPair(const pair<int,double> &p1, const pair<int,double> &p2) {
	return p1.second < p2.second;
}

static DisjointSetForest unconnectedNormalizedCuts(const WeightedGraph &graph, double stop) {
	// recursive call on each connected components of both subgraphs
	vector<int> inConnectedComponents;
	int nbCC;

	connectedComponents(graph, inConnectedComponents, &nbCC);

	vector<DisjointSetForest> partitions(nbCC);

	vector<WeightedGraph> components;
	vector<int> subgraphVertexIdx;

	inducedSubgraphs(graph, inConnectedComponents, nbCC, subgraphVertexIdx, components);

	for (int j = 0; j < nbCC; j++) {
		partitions[j] = normalizedCuts(components[j], stop);
	}

	// fuse the partitions of each connected component in the subgraph

	DisjointSetForest graphPartition(graph.numberOfVertices());

	fusePartitions(graph, inConnectedComponents, subgraphVertexIdx, partitions, graphPartition);

	return graphPartition;
}

/**
 * True iff the eigenvector is considered unstable, that is if it is a continuously
 * varying function with multiple best normalized cut points.
 *
 * @param eigenvector eigenvector of the laplacian matrix to check the stability
 * of.
 * @return true iff the eigenvector is considered unstable.
 */
static bool unstable(VectorXd eigenvector) {
	int numberOfBins = eigenvector.size() / min(eigenvector.size(), 10);
	VectorXi histogram = VectorXi::Zero(numberOfBins);
	double minVal = eigenvector.minCoeff();
	double maxVal = eigenvector.maxCoeff();

	for (int i = 0; i < eigenvector.size(); i++) {
		double val = eigenvector(i);
		double relPos = (val - minVal) / (maxVal - minVal);
		int bin = (int)((double)(numberOfBins - 1) * relPos);
		cout<<"bin = "<<bin<<endl;

		histogram(bin)++;
	}

	int maxBin = histogram.maxCoeff();
	int minBin = histogram.minCoeff();

	return ((double)minBin/(double)maxBin) > 0.06;
}

DisjointSetForest normalizedCuts(const WeightedGraph &graph, double stop) {
	assert(graph.numberOfVertices() >= 1);
	assert(connected(graph));

	// if the graph has only one vertex, it cannot be subdivided any further
	if (graph.numberOfVertices() == 1) {
		return DisjointSetForest(1);
	}
	// if the graph has only 2 vertices, there is only one valid bipartition
	if (graph.numberOfVertices() == 2) {
		return DisjointSetForest(2);
	}

	cout<<"computing normalized laplacian"<<endl;
	cout<<"graph has "<<graph.numberOfVertices()<<" vertices and "<<graph.getEdges().size()<<" edges"<<endl;
	VectorXd degrees;
	SparseMatrix<double> L = normalizedSparseLaplacian(graph, degrees);
	
	VectorXd evalues;
	MatrixXd evectors;

	symmetricSparseEigenSolver(L, "SA", 2, graph.numberOfVertices(), evalues, evectors);
	cout<<"computing D^-0.5"<<endl;
	// translate the computed eigemvector into a solution to the generalized
	// eigensystem.
	vector<Triplet<double> > tripletList;
	tripletList.reserve(graph.numberOfVertices());

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		if (degrees(i) == 0) {
			tripletList.push_back(Triplet<double>(i,i,0));
		} else {
			tripletList.push_back(Triplet<double>(i,i,1/sqrt(degrees(i))));
		}
	}

	SparseMatrix<double> invSqrtD(graph.numberOfVertices(), graph.numberOfVertices());

	invSqrtD.setFromTriplets(tripletList.begin(), tripletList.end());
	
	cout<<"computing the eigenvector in the non general eigensystem"<<endl;
	VectorXd eigenvector = invSqrtD * evectors.col(1);

	cout<<"checking stability of the eigenvector"<<endl;
	// checks stability of the eigenvector. If it is unstable, stop the recursion.
	if (unstable(eigenvector)) {
		cout<<"unstable, returning full segment"<<endl;
		DisjointSetForest wholeSegment(graph.numberOfVertices());

		for (int i = 1; i < graph.numberOfVertices(); i++) {
			wholeSegment.setUnion(0, i);
		}
	}

	//cout<<"evalues = "<<endl<<evalues<<endl;
	//cout<<"evectors = "<<endl<<evectors<<endl;

	vector<pair<int,double> > sortedWithIndex;

	sortedWithIndex.reserve(graph.numberOfVertices());

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		sortedWithIndex.push_back(pair<int,double>(i,eigenvector(i)));
	}

	sort(sortedWithIndex.begin(), sortedWithIndex.end(), compPair);

	VectorXd sortedEvec(graph.numberOfVertices());
	vector<int> sorting(graph.numberOfVertices());

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		sortedEvec(i) = sortedWithIndex[i].second;
		sorting[i] = sortedWithIndex[i].first;
	}

	vector<int> inSubgraph;

	cout<<"computing best normalized cut"<<endl;
	pair<int,double> bestCut = normalizedCutThreshold(graph, degrees, sortedEvec, sorting, inSubgraph);
	
	cout<<"best cut at "<<bestCut.first<<" of "<<graph.numberOfVertices()<<" with ratio "<<bestCut.second<<endl;

	// if the cut is good enough, compute bipartition
	cout<<"best ratio = "<<bestCut.second<<", stop = "<<stop<<endl;
	if (bestCut.second > stop) {
		cout<<"good enough cut, returning bipartition"<<endl;
		DisjointSetForest bipartition(graph.numberOfVertices());

		for (int i = 1; i <= bestCut.first; i++) {
			bipartition.setUnion(sorting[0], sorting[i]);
		}

		for (int j = bestCut.first + 2; j < graph.numberOfVertices(); j++) {
			bipartition.setUnion(sorting[bestCut.first + 1], sorting[j]);
		}

		return bipartition;
	} else {
		// otherwise, call the algorithm recursively on the connected components of
		// the subgraphs induced by each partition
		vector<WeightedGraph> subgraphs;
		vector<int> vertexIdx;

		cout<<"computing subgraphs"<<endl;
		inducedSubgraphs(graph, inSubgraph, 2, vertexIdx, subgraphs);

		for (int i = 0; i < 2; i++) {
			cout<<"subgraph "<<i<<": n = "<<subgraphs[i].numberOfVertices()<<", m = "<<subgraphs[i].getEdges().size()<<endl;
		}

		vector<DisjointSetForest> subgraphPartitions(2);

		for (int i = 0; i < 2; i++) {
			subgraphPartitions[i] = unconnectedNormalizedCuts(subgraphs[i], stop);
		}

		DisjointSetForest partition(graph.numberOfVertices());

		fusePartitions(graph, inSubgraph, vertexIdx, subgraphPartitions, partition);

		return partition;
	}
}

static double simpleKernel(const Mat &h1, const Mat &h2) {
	return gaussianKernel(1, 0.8, h1, h2);
}

static double radiusKernel(const Mat &h1, const Mat &h2) {
	return 
		gaussianKernel(1, 0.8, h1.colRange(2,5), h2.colRange(2,5)) *
		gaussianKernel(1, 0.8, h1.colRange(0,2), h2.colRange(0,2));
}

DisjointSetForest normalizedCutsSegmentation(const Mat_<Vec<uchar,3> > &image, const Mat_<float> &mask, double stop, int minCompSize) {
	cout<<"computing nearest neighbor graph"<<endl;
	WeightedGraph graph = radiusGraph(image, mask, 4, 2, radiusKernel, true);
	WeightedGraph grid = gridGraph(image, CONNECTIVITY_4, mask, simpleKernel, true);

	vector<int> vertexMap;

	cout<<"removing isolated vertices"<<endl;
	WeightedGraph connected = removeIsolatedVertices(graph, vertexMap);
	DisjointSetForest segmentationConn = unconnectedNormalizedCuts(connected, stop);
	cout<<"inserting isolated vertices"<<endl;
	DisjointSetForest segmentation = addIsolatedVertices(graph, segmentationConn, vertexMap);
	cout<<"before fusion: "<<segmentation.getNumberOfComponents()<<endl;
	segmentation.fuseSmallComponents(grid, minCompSize, mask);
	cout<<"after fusion: "<<segmentation.getNumberOfComponents()<<endl;

	return segmentation;
}
