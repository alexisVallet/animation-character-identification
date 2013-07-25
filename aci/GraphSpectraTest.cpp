#include "GraphSpectraTest.h"

static bool denseIsSymmetric(const Mat_<double> &toCheck) {
	if (toCheck.rows != toCheck.cols) {
		return false;
	}

	for (int i = 0; i < toCheck.rows; i++) {
		for (int j = i + 1; j < toCheck.rows; j++) {
			if (toCheck(i,j) != toCheck(j,i)) {
				return false;
			}
		}
	}

	return true;
}

static void testDenseLaplacian(WeightedGraph &testGraph) {
	Mat_<double> denseLaplacian = laplacian(testGraph);

	cout<<"testing symmetry"<<endl;
	assert(denseIsSymmetric(denseLaplacian));
	
	// checks positive semi-definiteness
	cout<<"testing positive semi-definiteness"<<endl;
	Mat_<double> eigenvalues;

	eigen(denseLaplacian, eigenvalues);

	for (int i = 0; i < testGraph.numberOfVertices(); i++) {
		assert(eigenvalues(i,0) > -10e-10);
	}

	// checks uninitialized buffer errors
	cout<<"testing uninitialized buffer errors"<<endl;
	Mat_<double> denseLaplacian2 = laplacian(testGraph);
	assert(denseLaplacian.rows == denseLaplacian2.rows && denseLaplacian.cols == denseLaplacian2.cols);

	for (int i = 0; i < denseLaplacian.rows; i++) {
		for (int j = 0; j < denseLaplacian.cols; j++) {
			assert(denseLaplacian(i,j) == denseLaplacian2(i,j));
		}
	}

	Mat_<double> eigenvalues2;

	eigen(denseLaplacian2, eigenvalues2);

	cout<<"ev1 = "<<eigenvalues<<endl;
	cout<<"ev2 = "<<eigenvalues2<<endl;

	for (int i = 0; i < testGraph.numberOfVertices(); i++) {
		assert(eigenvalues(i,0) == eigenvalues2(i,0));
	}
}

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
			if (!(abs(denseLaplacian(i,j) - sparseLapl.coeffRef(i,j)) <= 10E-8)) {
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
	set<std::tuple<int,int,float> > edges;

	for (int i = 0; i < nbEdges; i++) {
		int src = rand()%graph.numberOfVertices();
		int dst = rand()%(graph.numberOfVertices()-1);
		float weight = abs((float)rand() / RAND_MAX);

		if (src == dst) {
			dst++;
		}

		edges.insert(std::tuple<int,int,float>(src,dst,weight));
	}

	for (set<std::tuple<int,int,float> >::iterator it = edges.begin(); it != edges.end(); it++) {
		std::tuple<int,int,float> edge = *it;

		graph.addEdge(get<0>(edge), get<1>(edge), get<2>(edge));
		bidir.addEdge(get<0>(edge), get<1>(edge), get<2>(edge));
		bidir.addEdge(get<1>(edge), get<0>(edge), get<2>(edge));
	}
}

void testSparseEigenSolver(WeightedGraph &graph) {
	VectorXd degrees;

	SparseMatrix<double> L = normalizedSparseLaplacian(graph, true, degrees);
	MatrixXd dense(L);

	VectorXd actualEval;
	MatrixXd actualEvec;

	symmetricSparseEigenSolver(L, "SA", graph.numberOfVertices() - 1, graph.numberOfVertices(), actualEval, actualEvec);

	SelfAdjointEigenSolver<MatrixXd> solver(dense);

	VectorXd expectedEval = solver.eigenvalues();
	
	cout<<"expected evalues:"<<endl<<expectedEval<<endl;
	cout<<"actual:"<<endl<<actualEval<<endl;
	cout<<"expected evectors:"<<endl<<solver.eigenvectors()<<endl;
	cout<<"actual:"<<endl<<actualEvec<<endl;
}

static void testBFS() {
	WeightedGraph test(8);
	int edges[] = {
		0, 1,
		0, 2,
		1, 3,
		1, 4,
		1, 5,
		2, 5,
		2, 6,
		4, 7
	};
	int expected[] = {
		5, 1, 2, 0, 3, 4, 6, 7
	};

	for (int i = 0; i < 8; i++) {
		int src = edges[toRowMajor(2, 0, i)];
		int dst = edges[toRowMajor(2, 1, i)];

		test.addEdge(src, dst, 1);
		test.addEdge(dst, src, 1);
	}

	vector<int> actual = breadthFirstSearch(test, 5);

	for (int i = 0; i < actual.size(); i++) {
		cout<<actual[i]<<", ";
	}
	cout<<endl;

	assert(actual.size() == test.numberOfVertices());

	for (int i = 0; i < test.numberOfVertices(); i++) {
		assert(expected[i] == actual[i]);
	}
}

static void testPermuteVertices() {
	WeightedGraph test(8);
	int edges[] = {
		0, 1,
		0, 2,
		1, 3,
		1, 4,
		1, 5,
		2, 5,
		2, 6,
		4, 7
	};

	for (int i = 0; i < 8; i++) {
		int src = edges[toRowMajor(2, 0, i)];
		int dst = edges[toRowMajor(2, 1, i)];

		test.addEdge(src, dst, 1);
		test.addEdge(dst, src, 1);
	}

	vector<int> permutation = breadthFirstSearch(test, 5);
	WeightedGraph permuted = permuteVertices(test, permutation);

	for (int i = 0; i < permutation.size(); i++) {
		cout<<permutation[i]<<" becomes "<<i<<endl;
	}

	cout<<test<<endl;
	cout<<permuted<<endl;
}

void testGraphSpectra() {
	testBFS();
	testPermuteVertices();

	/*for (int i = 0; i < 100; i++) {
		WeightedGraph randomGraph(50);
		WeightedGraph bidir(50);

		randomBidirectional(randomGraph, bidir, 200);

		testDenseLaplacian(randomGraph);
	}*/
}
