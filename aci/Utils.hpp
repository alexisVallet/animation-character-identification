#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <utility>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

/**
 * Datatype indicating pixel connectivity.
 */
enum ConnectivityType {CONNECTIVITY_4 = 0, CONNECTIVITY_8 = 1};

/**
 * Converts coordinates in 2D array to row major format.
 */
int toRowMajor(int width, int x, int y);

/**
 * Converts a row major index to coordinates in a 2D array.
 */
pair<int,int> fromRowMajor(int width, int i);

/**
 * Loads a data set from a specific folder.
 */
void loadDataSet(char* folderName, char** charaNames, int nbCharas, int nbImagesPerChara, vector<Mat> &images, Mat &classes);

#endif
