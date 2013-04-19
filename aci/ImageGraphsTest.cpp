#include "ImageGraphsTest.h"

void testGridGraphBidirectional(Mat_<Vec<uchar,3> > &image, Mat_<float> &mask) {
	WeightedGraph grid = gridGraph(image, CONNECTIVITY_4, mask, true);
	WeightedGraph connectedGrid = removeIsolatedVertices(grid);
	set<pair<int,int> > edges;

	// checking that the graph representation is correctly bidirectional
	for (int i = 0; i < connectedGrid.getEdges().size(); i++) {
		Edge edge = connectedGrid.getEdges()[i];
		edges.insert(pair<int,int>(edge.source, edge.destination));
	}

	set<pair<int,int> >::iterator it;

	for (it = edges.begin(); it != edges.end(); it++) {
		pair<int,int> edge = *it;

		assert(edges.find(pair<int,int>(edge.second, edge.first)) != edges.end());
	}

	// checks that each vertex has at most 4 neighbors
	for (int i = 0; i < connectedGrid.numberOfVertices(); i++) {
		assert(connectedGrid.getAdjacencyList(i).size() <= 4);
	}

	Eigen::VectorXd degrees;

	// checks that its laplacian is symmetric
	Eigen::SparseMatrix<double> L = sparseLaplacian(connectedGrid, true, degrees);

	assert(symmetric(L));

	for (int i = 0; i < connectedGrid.numberOfVertices(); i++) {
		double sumWeights = 0;

		for (int j = 0; j < connectedGrid.getAdjacencyList(i).size(); j++) {
			sumWeights += connectedGrid.getAdjacencyList(i)[j].weight;
		}

		assert(abs(sumWeights - degrees(i)) <= 10E-8);
		assert(abs(degrees(i) - L.coeffRef(i,i)) <= 10E-8);
	}

	// checks that the all 1 vector is an eigenvector of eigenvalue 0 of the laplacian
	Eigen::VectorXd evector = Eigen::VectorXd::Ones(connectedGrid.numberOfVertices());

	assert((L*evector).norm() <= 10E-8);
}

void testImageGraphs() {
	cout<<"opening image"<<endl;
	Mat_<Vec<uchar,3> > testImage = imread("../test/dataset/asuka_a.png");
	cout<<"opening mask"<<endl;
	Mat_<Vec<uchar, 3> > rgbMask = imread("../test/dataset/asuka_a.png-mask.png");
	cout<<"processing mask"<<endl;
	vector<Mat_<uchar> > maskChannels;
	split(rgbMask, maskChannels);
	Mat_<float> mask = Mat_<float>::ones(rgbMask.rows, rgbMask.cols) - (Mat_<float>(maskChannels[0]) / 255);

	cout<<"testing bidirectional grid graph"<<endl;
	testGridGraphBidirectional(testImage, mask);
	cout<<"passed"<<endl;
}