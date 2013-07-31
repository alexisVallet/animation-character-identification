#pragma once

#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <opencv2/opencv.hpp>

#include "Utils.hpp"
#include "CSVIterator.h"

using namespace std;
using namespace cv;
using namespace Eigen;

/**
 * Loads a data set from a specific folder.
 *
 * @param folderName name of folder containing image files, mask files and
 * face positions file.
 * @param charaNames NULL terminated array containing the name of each
 * character, expected to be the prefix in the file name.
 * @param nbImagesPerChara number of images for each character.
 * @param images output vector of character images along with mask and
 * and face position.
 * @param classes class label associated to each character image.
 */
void loadDataSet(char* folderName, char** charaNames, int nbImagesPerChara, vector<std::tuple<Mat_<Vec<uchar,3> >,Mat_<float> > > &images, Mat_<int> &classes);
