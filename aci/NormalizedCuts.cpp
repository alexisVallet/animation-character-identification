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
	inSubgraph = vector<int>(graph.numberOfVertices(), 1);
	inSubgraph[sorting[0]] = 0;
	double cutAB = degrees(sorting[0]);
	double assocAV = degrees(sorting[0]);
	double assocBV = degrees.sum() - degrees(sorting[0]);
	double bestRatio = cutAB/assocAV + cutAB/assocBV;
	int bestCut = 0;

	for (int i = 1; i < graph.numberOfVertices(); i++) {
		assocAV += degrees(sorting[i]);
		assocBV -= degrees(sorting[i]);

		double internalWeights = 0;

		for (int j = 0; j < graph.getAdjacencyList(sorting[i]).size() - 1; j++) { // -1 because we don't want the entire graph
			HalfEdge edge = graph.getAdjacencyList(sorting[i])[j];

			if (inSubgraph[edge.destination] == 0) {
				internalWeights += edge.weight;
			}
		}

		cutAB += degrees(sorting[i]) - 2 * internalWeights;

		double newRatio = cutAB/assocAV + cutAB/assocBV;

		if (newRatio < bestRatio) {
			bestRatio = newRatio;
			bestCut = i;
		}

		inSubgraph[sorting[i]] = 0;
	}

	return pair<int,double>(bestCut,bestRatio);
}

static bool compPair(const pair<int,double> &p1, const pair<int,double> &p2) {
	return p1.second < p2.second;
}

DisjointSetForest normalizedCuts(const WeightedGraph &graph, double stop) {
	assert(connected(graph));
	cout<<"computing normalized laplacian"<<endl;
	VectorXd degrees;
	SparseMatrix<double> L = normalizedSparseLaplacian(graph, degrees);
	
	VectorXd evalues;
	MatrixXd evectors;

	symmetricSparseEigenSolver(L, "SA", 2, 3 * graph.numberOfVertices(), evalues, evectors);

	cout<<"evalues = "<<endl<<evalues<<endl;
	//cout<<"evectors = "<<endl<<evectors<<endl;

	vector<pair<int,double> > sortedWithIndex;

	sortedWithIndex.reserve(graph.numberOfVertices());

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		sortedWithIndex.push_back(pair<int,double>(i,evectors(i,1)));
	}

	sort(sortedWithIndex.begin(), sortedWithIndex.end(), compPair);

	VectorXd sortedEvec(graph.numberOfVertices());
	vector<int> sorting(graph.numberOfVertices());

	for (int i = 0; i < graph.numberOfVertices(); i++) {
		sortedEvec(i) = sortedWithIndex[i].second;
		sorting[i] = sortedWithIndex[i].first;
	}

	vector<int> inSubgraph;

	pair<int,double> bestCut = normalizedCutThreshold(graph, degrees, sortedEvec, sorting, inSubgraph);

	cout<<"best cut at "<<bestCut.first<<" of "<<graph.numberOfVertices()<<" with ratio "<<bestCut.second<<endl;

	// if the cut is good enough, compute bipartition
	if (bestCut.second <= stop) {
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

		inducedSubgraphs(graph, inSubgraph, 2, vertexIdx, subgraphs);

		vector<DisjointSetForest> subgraphPartitions(2);

		for (int i = 0; i < 2; i++) {
			// recursive call on each connected components of both subgraphs
			vector<int> inConnectedComponents;
			int nbCC;

			connectedComponents(subgraphs[i], inConnectedComponents, &nbCC);

			vector<DisjointSetForest> partitions(nbCC);

			vector<WeightedGraph> components;
			vector<int> subgraphVertexIdx;

			inducedSubgraphs(subgraphs[i], inConnectedComponents, nbCC, subgraphVertexIdx, components);

			for (int j = 0; j < nbCC; j++) {
				partitions[j] = normalizedCuts(components[j], stop);
			}

			// fuse the partitions of each connected component in the subgraph

			subgraphPartitions[i] = DisjointSetForest(subgraphs[i].numberOfVertices());

			fusePartitions(subgraphs[i], inConnectedComponents, subgraphVertexIdx, partitions, subgraphPartitions[i]);
		}

		DisjointSetForest partition(graph.numberOfVertices());

		fusePartitions(graph, inSubgraph, vertexIdx, subgraphPartitions, partition);

		return partition;
	}
}

DisjointSetForest normalizedCutsSegmentation(const Mat_<Vec<uchar,3> > &image, const Mat_<float> &mask, double stop) {
	WeightedGraph &graph = nearestNeighborGraph(image, 5);

	vector<int> vertexMap;

	WeightedGraph connected = removeIsolatedVertices(graph, vertexMap);
	DisjointSetForest segmentationConn = normalizedCuts(connected, 0.1);
	DisjointSetForest segmentation = addIsolatedVertices(graph, segmentationConn, vertexMap);

	return segmentation;
}