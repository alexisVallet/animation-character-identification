#include "Kernels.h"

double euclidDistance(const Mat &h1, const Mat &h2) {
	return norm(h1, h2);
}

static int uniformMap(int binsPerChannel, unsigned char channelValue) {
	return floor(((float)channelValue/255.0)*(binsPerChannel-1));
}

double dotProductKernel(const Mat &h1, const Mat &h2) {
	return h1.dot(h2);
}

double gaussianKernel(float sigma, const Mat &h1, const Mat &h2) {
	return exp(-pow(norm(h1, h2), 2) / ( 2 * pow(sigma, 2)));
}

double khi2Kernel(int binsPerChannel, float lambda, float mu, float gamma, int area1, const Mat &h1, int area2, const Mat &h2) {
	double d2 = 0;

	for (int r = 0; r < binsPerChannel; r++) {
		for (int g = 0; g < binsPerChannel; g++) {
			for (int b = 0; b < binsPerChannel; b++) {
				int pi = h1.at<int>(r,g,b);
				int qi = h2.at<int>(r,g,b);

				if (pi + qi != 0) {
					double toAdd = pow((float)(pi-qi),2) / (float)(pi + qi);
					
					d2 += toAdd;
				}
			}
		}
	}

	double result = lambda * pow(area1, gamma) * pow(area2, gamma) * exp(-mu * d2);

	return result;
}

void colorHistogramLabels(
    Mat_<Vec<uchar,3> > &image, 
    DisjointSetForest &segmentation, 
    LabeledGraph<Mat> &segmentationGraph,
    int binsPerChannel) {
	int numberOfComponents = segmentation.getNumberOfComponents();
	map<int,int> rootIndexes = segmentation.getRootIndexes();
	vector<Mat> histograms(numberOfComponents);
	int dims[3] = {binsPerChannel, binsPerChannel, binsPerChannel};
	for (int i = 0; i < numberOfComponents; i++) {
		histograms[i] = Mat(3, dims, CV_32S);

		for (int j = 0; j < dims[0]; j++) {
			for (int k = 0; k < dims[1]; k++) {
				for (int l = 0; l < dims[2]; l++) {
					histograms[i].at<int>(j,k,l) = 0;
				}
			}
		}
	}

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			int pixComp =
				rootIndexes[segmentation.find(toRowMajor(image.cols,j,i))];
			Vec<uchar,3> pixColor = image(i,j);
			int r = uniformMap(binsPerChannel,pixColor[0]);
			int g = uniformMap(binsPerChannel,pixColor[1]);
			int b = uniformMap(binsPerChannel,pixColor[2]);
			histograms[pixComp].at<int>(r, g, b)++;
		}
	}

	for (int i = 0; i < numberOfComponents; i++) {
		segmentationGraph.addLabel(i, histograms[i]);
	}
}

void averageColorLabels(const Mat_<Vec3b> &image, const Mat_<float> &mask, DisjointSetForest &segmentation, LabeledGraph<Mat> &segmentationGraph) {
	assert(image.rows == mask.rows && image.cols == mask.cols);
	vector<Vec3f> averageColor;
	averageColor.reserve(segmentation.getNumberOfComponents());

	for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
		averageColor.push_back(Vec3f(0,0,0));
	}
	map<int,int> rootIndexes = segmentation.getRootIndexes();

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (mask(i,j) > 0) {
				int root = rootIndexes[segmentation.find(toRowMajor(image.cols, j, i))];
				Vec3f pixColor = Vec3f(image(i,j));

				averageColor[root] += pixColor / (float)segmentation.getComponentSize(root);
			}
		}
	}

	for (int i = 0; i < segmentation.getNumberOfComponents(); i++) {
		segmentationGraph.addLabel(i, Mat(averageColor[i]));
	}
}
