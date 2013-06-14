#include "TWKSpectralClusteringTest.h"

#define TWK_NBCLASSES 12

void testTWKSpectralClustering(ofstream &out, int depth, int arity, SpectralClusteringType clusteringType, const vector<pair<Mat_<Vec3b>, Mat_<float> > > &dataSet, const vector<WeightedGraph> &segGraphs, vector<DisjointSetForest> &segmentations) {
	cout<<"labeling"<<endl;
	// labeling
	vector<LabeledGraph<Matx<float,4,1> > > labeled;
	TWBasisKernel twKernel(5, 0);

	for (int i = 0; i < (int)dataSet.size(); i++) {	
		LabeledGraph<Matx<float,4,1> > labeledGraph;

		twKernel.getLabeling()(dataSet[i].first, dataSet[i].second, segmentations[i], segGraphs[i], labeledGraph);

		labeled.push_back(labeledGraph);
	}

	cout<<"embedding"<<endl;
	TWKSpectralClustering<float,4,1> clustering(dataSet, (MatKernel<float,4,1>*)&twKernel, depth, arity, clusteringType);
	MatrixXd embeddings;

	clustering.embed(segmentations, labeled, 3, embeddings);

	eigenMatToCsv(embeddings, out);
}
