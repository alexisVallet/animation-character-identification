#include "PrincipalAnglesClassifierTest.h"

void testPrincipalAnglesClassifier() {
	WeightedGraph g(2), h(3), i(3);

	g.addEdge(0,1,1);

	h.addEdge(0,1,1);
	h.addEdge(0,2,1);
	h.addEdge(1,2,1);

	i.addEdge(0,1,1);
	i.addEdge(1,2,1);

	vector<pair<WeightedGraph,int> > dataSet;
	dataSet.reserve(4);
	pair<WeightedGraph,int> p1(g,0), p2(h,1);
	dataSet.push_back(p1);
	dataSet.push_back(p2);

	PrincipalAnglesClassifier classifier(
		PrincipalAnglesClassifier::SMALLEST_ANGLE,
		eigNormalizedLaplacian,
		true);

	classifier.train(dataSet);
	cout<<"predicting g"<<endl;
	int l1 = classifier.predict(g);
	cout<<"predicting h"<<endl;
	int l2 = classifier.predict(h);
	cout<<"predicting i"<<endl;
	int l3 = classifier.predict(i);

	cout<<"l1 = "<<l1<<endl;
	cout<<"l2 = "<<l2<<endl;
	cout<<"l3 = "<<l3<<endl;
}
