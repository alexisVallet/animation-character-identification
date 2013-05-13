#include "SegmentAttributesTest.h"

void testSegmentAttributes() {
	char *folder = "../test/dataset/";
	char *names[] = {"asuka", "amuro"};
	vector<pair<Mat_<Vec<uchar,3> >,Mat_<float> > > dataSet;
	Mat classes;

	loadDataSet(folder, names, 2, 5, dataSet, classes);

	for (int i = 0; i < dataSet.size(); i++) {
		// check for seemingly random differences which may be caused by uninitialized
		// buffers.
		WeightedGraph segGraph1 = computeGraphFrom(dataSet[i].first, dataSet[i].second);
		WeightedGraph segGraph2 = computeGraphFrom(dataSet[i].first, dataSet[i].second);

		cout<<"checking vertex set size"<<endl;
		assert(segGraph1.numberOfVertices() == segGraph2.numberOfVertices());
		cout<<"checking edge set size"<<endl;
		assert(segGraph1.getEdges().size() == segGraph2.getEdges().size());

		cout<<"checking edges are identical"<<endl;
		for (int i = 0; i < segGraph1.getEdges().size(); i++) {
			Edge edge1 = segGraph1.getEdges()[i];
			Edge edge2 = segGraph2.getEdges()[i];

			assert(edge1.source == edge2.source);
			assert(edge1.destination == edge2.destination);
			assert(edge1.weight == edge2.weight);
		}
	}
}