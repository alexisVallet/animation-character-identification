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

void testSparseEigenSolver(WeightedGraph &graph) {
	SparseMatrix<double> L = normalizedSparseLaplacian(graph);
	MatrixXd dense(L);

	VectorXd actualEval;
	MatrixXd actualEvec;

	symmetricSparseEigenSolver(L, "SA", graph.numberOfVertices() - 1, actualEval, actualEvec);

	SelfAdjointEigenSolver<MatrixXd> solver(dense);

	VectorXd expectedEval = solver.eigenvalues();
	
	cout<<"expected evalues:"<<endl<<expectedEval<<endl;
	cout<<"actual:"<<endl<<actualEval<<endl;
	cout<<"expected evectors:"<<endl<<solver.eigenvectors()<<endl;
	cout<<"actual:"<<endl<<actualEvec<<endl;
}

void testGraphSpectra() {
	WeightedGraph graph(4);

	int edges[4][2] = {{0,1},{0,2},{0,3},{2,3}};

	for (int i = 0; i < 4; i++) {
		graph.addEdge(edges[i][0], edges[i][1], 1);
		graph.addEdge(edges[i][1], edges[i][0], 1);
	}

	testSparseEigenSolver(graph);
}