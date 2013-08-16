#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <limits>

#include "Utils.hpp"
#include "KuwaharaFilter.h"
#include "ImageGraphs.h"

#define DEFAULT_KUWAHARA_HALFSIZE 5
#define DEFAULT_MAX_NB_PIXELS 15000

using namespace std;
using namespace cv;

/**
 * Pre-process an animation character image for segmentation. Proceeds to, in order:
 * - resize the image and mask so the number of non masked pixel is below a maximum,
 *   for performance.
 * - only keep the largest 4-connected component in the mask, so there is only one
 * - equalize the color histogram by Hue for better color repartition.
 * - apply Kuwahara filter for outline removal and more homogenous areas.
 * - convert to Lab color space so segmentation is closer to human perception.
 *
 * @param rawImage BGR image of an animation character, as returned from imread for instance.
 * @param rawMask mask indicating which pixels to take into account into the raw image,
 * 0 to ignore and 1 to take into account.
 * @param processedImage output pre-processed image.
 * @param processedMask output pre-processed mask.
 * @param kuwaharaHalfsize window halfsize for the Kuwahara filtering algorithm.
 * @param maxNbPixels maximum allowed number of non-masked pixels for resizing.
 */
void preProcessing(const Mat_<Vec3b> &rawImage, const Mat_<float> &rawMask, Mat_<Vec3b> &processedImage, Mat_<float> &processedMask, const Mat_<Vec3b> &manualSegmentation = Mat_<Vec3b>(), Mat_<Vec3b> &processedSegmentation = Mat_<Vec3b>(), int kuwaharaHalfsize = DEFAULT_KUWAHARA_HALFSIZE, int maxNbPixels = DEFAULT_MAX_NB_PIXELS);
