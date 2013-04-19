#include "GraphSpectraTest.h"

static void testSparseLaplacian(WeightedGraph &testGraph) {
	Mat_<double> denseLaplacian = laplacian(testGraph);
	Eigen::VectorXd degrees = Eigen::VectorXd::Zero(testGraph.numberOfVertices());
	Eigen::SparseMatrix<double> sparseLapl = sparseLaplacian(testGraph, false, degrees);

	for (int i = 0; i < testGraph.numberOfVertices(); i++) {
		for (int j = 0; j < testGraph.numberOfVertices(); j++) {
			assert(abs(denseLaplacian(i,j) - sparseLapl.coeffRef(i,j)) <= 10E-8);
		}
	}
}

static void testSparseLaplacianBidirectional(WeightedGraph &testGraph, WeightedGraph &bidirectional) {
	// checks that the sparse laplacian is equal to the dense one
	Mat_<double> denseLaplacian = laplacian(testGraph);
	Eigen::VectorXd degrees = Eigen::VectorXd::Zero(testGraph.numberOfVertices());
	Eigen::SparseMatrix<double> sparseLapl = sparseLaplacian(bidirectional, true, degrees);

	for (int i = 0; i < testGraph.numberOfVertices(); i++) {
		for (int j = 0; j < testGraph.numberOfVertices(); j++) {
			if (!abs(denseLaplacian(i,j) - sparseLapl.coeffRef(i,j)) <= 10E-8) {
				cout<<testGraph<<endl;
				cout<<bidirectional<<endl;
				cout<<"not equal at coefficient ("<<i<<","<<j<<")"<<endl;
				cout<<"dense:"<<endl<<denseLaplacian<<endl;
				cout<<"sparse:"<<endl<<sparseLapl<<endl;
				assert(false);
			}
		}
	}

	// checks that the sparse laplacian is positive definite when we remove the line/column
	// of the vertex of highest degree.
	int ground;

	degrees.maxCoeff(&ground);

	Eigen::SparseMatrix<double> L0(sparseLapl.rows(), sparseLapl.cols());

	removeLineCol(sparseLapl, ground, L0);

	assert(positiveDefinite(L0));
}

void randomBidirectional(WeightedGraph& graph, WeightedGraph& bidir, int nbEdges) {
	set<std::tuple<int,int,double> > edges;

	for (int i = 0; i < nbEdges; i++) {
		int src = rand()%graph.numberOfVertices();
		int dst = rand()%(graph.numberOfVertices()-1);
		double weight = abs((double)rand() / RAND_MAX);

		if (src == dst) {
			dst++;
		}

		edges.insert(std::tuple<int,int,double>(src,dst,weight));
	}

	for (set<std::tuple<int,int,double> >::iterator it = edges.begin(); it != edges.end(); it++) {
		std::tuple<int,int,double> edge = *it;

		graph.addEdge(get<0>(edge), get<1>(edge), get<2>(edge));
		bidir.addEdge(get<0>(edge), get<1>(edge), get<2>(edge));
		bidir.addEdge(get<1>(edge), get<0>(edge), get<2>(edge));
	}
}

void testGraphSpectra() {
	WeightedGraph testGraph(4);

	testGraph.addEdge(0,1,1);
	testGraph.addEdge(0,3,5);
	testGraph.addEdge(0,2,5);
	testGraph.addEdge(2,3,5);

	cout<<"testing function sparseLaplacian"<<endl;
	testSparseLaplacian(testGraph);
	
	// tests sparse laplacian from bidirectional graph for randomly
	// generated large graphs without loops
	for (int i = 0; i < 100; i++) {
		WeightedGraph graph(10);
		WeightedGraph biDir(10);

		randomBidirectional(graph, biDir, 30);

		testSparseLaplacianBidirectional(graph, biDir);
	}
}