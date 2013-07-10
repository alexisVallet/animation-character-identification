#include "PrincipalAnglesClassifierTest.h"

void testPrincipalAnglesClassifier() {
	WeightedGraph g(2), h(3);

	g.addEdge(0,1,1);

	h.addEdge(0,1,1);
	h.addEdge(0,2,1);
	h.addEdge(1,2,1);

	KNearestModel statModel(1);

	PrincipalAnglesClassifier classifier(&statModel, eigNormalizedLaplacian, false, true, 3);

	vector<pair<WeightedGraph,int> > dataSet;
	dataSet.reserve(4);
	pair<WeightedGraph,int> p1(g,0), p2(h,1);
	dataSet.push_back(p1);
	dataSet.push_back(p1);
	dataSet.push_back(p2);
	dataSet.push_back(p2);

	classifier.leaveOneOutRecognitionRate(dataSet);
}
