#include "ImageGraphs.h"

#define MIN_EDGE_WEIGHT 0

WeightedGraph gridGraph(const Mat_<Vec<uchar,3> > &image, ConnectivityType connectivity, Mat_<float> mask, MatKernel simFunc, bool bidirectional) {
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
						
						float weight = simFunc(Mat(centerIntensity), Mat(neighborIntensity));

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

Mat pixelFeatures(const Mat_<Vec<uchar,3> > &image, const Mat_<float> &mask) {
	int nonZeros = countNonZero(mask);
	// computes the set of features of the image
	Mat features(nonZeros, 5, CV_32F);
	int index = 0;
	
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (mask(i,j) > 0) {
				Vec<uchar,3> color = image(i,j);

				features.at<float>(index,0) = (float)i;
				features.at<float>(index,1) = (float)j;
				features.at<float>(index,2) = color[0];
				features.at<float>(index,3) = color[1];
				features.at<float>(index,4) = color[2];
				index++;
			}
		}
	}

	cout<<"expected = "<<index<<", actual = "<<nonZeros<<endl;

	return features;
}

WeightedGraph kNearestGraph(const Mat_<Vec<uchar,3> > &image, const Mat_<float> mask, int k, MatKernel simFunc, bool bidirectional) {
	cout<<"computing features"<<endl;
	Mat features = pixelFeatures(image, mask);
	cout<<"computing index"<<endl;
	flann::Index flannIndex(features, flann::KMeansIndexParams(16, 5));
	WeightedGraph nnGraph(image.rows * image.cols);
	set<pair<int,int> > edges;

	// for each feature, determine the k nearest neighbors and add them
	// as edges to the graph.
	for (int i = 0; i < features.rows; i++) {
		vector<int> indices(k + 1);
		vector<float> distances(k + 1);

		flannIndex.knnSearch(features.row(i), indices, distances, k + 1);
		int srcI = (int)features.at<float>(i, 0);
		int srcJ = (int)features.at<float>(i, 1);
		int source = toRowMajor(image.cols, srcJ, srcI);

		for (int j = 0; j < k + 1; j++) {
			if (indices[j] != i) {
				int dstI = (int)features.at<float>(indices[j], 0);
				int dstJ = (int)features.at<float>(indices[j], 1);
				int destination = toRowMajor(image.cols, dstJ, dstI);
				
				int first = min(source, destination);
				int second = max(source, destination);

				if (edges.find(pair<int,int>(first, second)) == edges.end()) {
					double weight = simFunc(features.row(i), features.row(indices[j])) + MIN_EDGE_WEIGHT;

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

WeightedGraph radiusGraph(const Mat_<Vec3b> &image, const Mat_<float> &mask, int k, double r, MatKernel simFunc, bool bidirectional) {
	Mat features = pixelFeatures(image, mask);
	// copying data because flann requires a continuous array.
	Mat positions = features.colRange(0, 2).clone();

	cout<<"initializing index"<<endl;
	flann::Index flannIndex(positions, flann::KMeansIndexParams(16, 5));

	WeightedGraph nnGraph(image.rows * image.cols);
	set<pair<int,int> > edges;

	// for each feature, determine the k nearest neighbors and add them
	// as edges to the graph.
	for (int i = 0; i < features.rows; i++) {
		vector<int> indices(k + 1);
		vector<float> distances(k + 1);

		flannIndex.radiusSearch(positions.row(i).colRange(0,2), indices, distances, r, k + 1);
		int srcI = (int)features.at<float>(i, 0);
		int srcJ = (int)features.at<float>(i, 1);
		int source = toRowMajor(image.cols, srcJ, srcI);

		for (int j = 0; j < k + 1; j++) {
			if (indices[j] != i) {
				int dstI = (int)features.at<float>(indices[j], 0);
				int dstJ = (int)features.at<float>(indices[j], 1);
				int destination = toRowMajor(image.cols, dstJ, dstI);
				
				int first = min(source, destination);
				int second = max(source, destination);

				if (edges.find(pair<int,int>(first, second)) == edges.end()) {
					double weight = simFunc(features.row(i), features.row(indices[j])) + MIN_EDGE_WEIGHT;

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
