#include "ImageGraphs.h"

#define MIN_EDGE_WEIGHT 5

static float euclDist(Vec<uchar,3> v1, Vec<uchar,3> v2) {
  float dr = ((float)v1[0]) - ((float)v2[0]);
  float dg = ((float)v1[1]) - ((float)v2[1]);
  float db = ((float)v1[2]) - ((float)v2[2]);

  return sqrt(dr*dr + dg*dg + db*db);
}

WeightedGraph gridGraph(Mat_<Vec<uchar,3> > &image, ConnectivityType connectivity, Mat_<float> mask, bool bidirectional) {
	assert(image.size() == mask.size());
	WeightedGraph grid(image.cols*image.rows, 4);
	// indicates neigbor positions depending on connectivity
	int numberOfNeighbors[2] = {2, 4};
	int colOffsets[2][4] = {{0, 1, 0, 0}, {-1, 0, 1, 1}};
	int rowOffsets[2][4] = {{1, 0, 0, 0}, { 1, 1, 1, 0}};

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (mask(i,j) >= 0.5) {
				int centerIndex = toRowMajor(image.cols, j,i);
				assert(centerIndex >= 0 && centerIndex < grid.numberOfVertices());
				Vec<uchar,3> centerIntensity = image(i,j);
      
				for (int n = 0; n < numberOfNeighbors[connectivity]; n++) {
					int neighborRow = i + rowOffsets[connectivity][n];
					int neighborCol = j + colOffsets[connectivity][n];
	
					if (neighborRow >= 0 && neighborRow < image.rows &&
						neighborCol >= 0 && neighborCol < image.cols &&
						mask(neighborRow, neighborCol) >= 0.5) {
						int neighborIndex = toRowMajor(image.cols, neighborCol, neighborRow);
						Vec<uchar,3> neighborIntensity = image(neighborRow, neighborCol);
	  
						assert(neighborIndex >= 0 && neighborIndex < grid.numberOfVertices());
						
						float weight = euclDist(centerIntensity, neighborIntensity);

						grid.addEdge(centerIndex, neighborIndex, weight + MIN_EDGE_WEIGHT);

						if (bidirectional) {
							grid.addEdge(neighborIndex, centerIndex, weight + MIN_EDGE_WEIGHT);
						}
					}
				}
			}
		}
	}

	return grid;
}

WeightedGraph nearestNeighborGraph(const Mat_<Vec<uchar,3> > &image, const Mat_<float> mask, int k) {
	// computes the set of features of the image
	Mat features(countNonZero(mask), 5, CV_32F);
	int index = 0;
	
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (mask(i,j) > 0.5) {
				Vec<uchar,3> color = image(i,j);

				features.at<float>(index,0) = (float)j;
				features.at<float>(index,1) = (float)i;
				features.at<float>(index,2) = color[0];
				features.at<float>(index,3) = color[1];
				features.at<float>(index,4) = color[2];
				index++;
			}
		}
	}

	flann::Index flannIndex(features, flann::KMeansIndexParams(16, 5));
	WeightedGraph nnGraph(image.rows * image.cols);

	// for each feature, determine the k nearest neighbors and add them
	// as edges to the graph.
	for (int i = 0; i < features.rows; i++) {
		vector<int> indices(k);
		vector<float> distances(k);
		set<pair<int,int>> edges;

		flannIndex.knnSearch(features.row(i), indices, distances, k);

		for (int j = 0; j < k; j++) {
			// if there isn't already an edge the other way, add an edge
			if (edges.find(pair<int,int>(indices[j], i)) == edges.end()) {
				nnGraph.addEdge(i, indices[j], distances[j]);
				edges.insert(pair<int,int>(i, indices[j]));
			}
		}
	}

	return nnGraph;
}
