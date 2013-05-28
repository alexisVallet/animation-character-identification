#include "TreeWalkKernelTest.h"

#define TWK_SIGMA ((double)255)

static double testKernel(int area1, const Vec3b &c1, int area2, const Vec3b &c2) {
	return exp(-pow(norm(Vec3f(c1), Vec3f(c2)), 2) / pow(TWK_SIGMA, 2));
}

void treeWalkKernelTest() {
	LabeledGraph<Vec3b> graph1(4,3);
	DisjointSetForest seg1(4);
	vector<Vec2f> embedding1;

	graph1.addLabel(0, Vec3b(255,0,0));
	graph1.addLabel(1, Vec3b(0,255,0));
	graph1.addLabel(2, Vec3b(0,0,255));
	graph1.addLabel(3, Vec3b(255,0,255));

	embedding1.push_back(Vec2f(1,1));
	embedding1.push_back(Vec2f(0,0));
	embedding1.push_back(Vec2f(2,0));
	embedding1.push_back(Vec2f(1,-1));

	for (int i = 0; i < 4; i++) {
		for (int j = i + 1; j < 4; j++) {
			graph1.addEdge(i,j,1);
			graph1.addEdge(j,i,1);
		}
	}

	LabeledGraph<Vec3b> graph2(2,1);
	DisjointSetForest seg2(2);
	vector<Vec2f> embedding2;

	graph2.addLabel(0, Vec3b(255,255,0));
	graph2.addLabel(1, Vec3b(0,255,255));

	embedding2.push_back(Vec2f(0,1));
	embedding2.push_back(Vec2f(1,0));

	graph2.addEdge(0,1,1);
	graph2.addEdge(1,0,1);

	double res = treeWalkKernel<Vec3b>(testKernel, 2, 2, seg1, graph1, embedding1, seg2, graph2, embedding2);

	cout<<"res = "<<res<<endl;
}