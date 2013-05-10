#include "ImageGraphs.h"

#define MIN_EDGE_WEIGHT 0

WeightedGraph gridGraph(const Mat_<Vec<uchar,3> > &image, ConnectivityType connectivity, Mat_<float> mask, double (*simFunc)(const Mat&, const Mat&), bool bidirectional) {
	assert(image.rows == mask.rows && image.cols == mask.cols);
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
						
						float weight = (float)simFunc(Mat(centerIntensity), Mat(neighborIntensity));

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

Mat pixelFeatures(const Mat_<Vec<uchar,3> > &image, const Mat_<float> &mask, vector<int> &indexToVertex) {
	int nonZeros = countNonZero(mask);
	// computes the set of features of the image
	Mat features(nonZeros, 5, CV_32F);
	int index = 0;
	indexToVertex = vector<int>(nonZeros,-1);
	
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (mask(i,j) > 0) {
				Vec<uchar,3> color = image(i,j);

				// computing the feature vector, normalizing coordinates between 0 and 1
				features.at<float>(index,0) = (float)i / (float)image.rows;
				features.at<float>(index,1) = (float)j / (float)image.cols;
				features.at<float>(index,2) = (float)color[0] / (float)255.;
				features.at<float>(index,3) = (float)color[1] / (float)255.;
				features.at<float>(index,4) = (float)color[2] / (float)255.;

				indexToVertex[index] = toRowMajor(image.cols, j, i);

				index++;
			}
		}
	}

	return features;
}

WeightedGraph kNearestGraph(const Mat_<Vec<uchar,3> > &image, const Mat_<float> mask, int k, double (*simFunc)(const Mat&, const Mat&), bool bidirectional) {
	vector<int> indexToVertex;
	Mat features = pixelFeatures(image, mask, indexToVertex);
	flann::Index flannIndex(features, flann::KMeansIndexParams(16, 5));
	WeightedGraph nnGraph(image.rows * image.cols);
	set<pair<int,int> > edges;

	// for each feature, determine the k nearest neighbors and add them
	// as edges to the graph.
	for (int i = 0; i < features.rows; i++) {
		vector<int> indices(k + 1);
		vector<float> distances(k + 1);

		flannIndex.knnSearch(features.row(i), indices, distances, k + 1);
		int source = indexToVertex[i];

		for (int j = 0; j < k + 1; j++) {
			if (indices[j] != i) {
				int destination = indexToVertex[indices[j]];
				
				int first = min(source, destination);
				int second = max(source, destination);

				if (edges.find(pair<int,int>(first, second)) == edges.end()) {
					float weight = (float)simFunc(features.row(i), features.row(indices[j])) + MIN_EDGE_WEIGHT;

					edges.insert(pair<int,int>(first, second));
					nnGraph.addEdge(source, destination, weight);

					if (bidirectional) {
						nnGraph.addEdge(destination, source, weight);
					}
				}
			}
		}
	}

	return nnGraph;
}