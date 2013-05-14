#include "PreProcessing.h"

void preProcessing(const Mat_<Vec3b> &rawImage, const Mat_<float> &rawMask, Mat_<Vec3b> &processedImage, Mat_<float> &processedMask, int kuwaharaHalfsize, int maxNbPixels) {
	assert(kuwaharaHalfsize <= (numeric_limits<uchar>::max() - 1) / 2);
	Mat_<Vec3b> resized;

	resizeImage(rawImage, rawMask, resized, processedMask, maxNbPixels);

	Mat_<Vec3b> equalized;
	
	equalizeColorHistogram(resized, processedMask, equalized);

	Mat_<Vec3b> filtered;

	KuwaharaFilter(equalized, filtered, 2 * kuwaharaHalfsize + 1);

	cvtColor(filtered, processedImage, CV_BGR2Lab);
}
