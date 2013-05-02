/** @file */
#pragma once

#include <utility>
#include <opencv2/opencv.hpp>
#include <Eigen/Sparse>

using namespace std;
using namespace cv;

typedef double (*MatKernel)(const Mat&, const Mat&);

/**
 * Datatype indicating pixel connectivity.
 */
enum ConnectivityType {CONNECTIVITY_4 = 0, CONNECTIVITY_8 = 1};

/**
 * Converts coordinates in 2D array to row major format.
 */
int toRowMajor(int width, int x, int y);

/**
 * Converts coordinates in 2D array to column major format.
 */
int toColumnMajor(int rows, int i, int j);

/**
 * Converts a row major index to coordinates in a 2D array.
 */
pair<int,int> fromRowMajor(int width, int i);

/**
 * Loads a data set from a specific folder.
 */
void loadDataSet(char* folderName, char** charaNames, int nbCharas, int nbImagesPerChara, vector<pair<Mat_<Vec<uchar,3> >,Mat_<float> > > &images, Mat &classes);

/**
 * Multiplies a sparse n by n matrix by a dense n by 1 column vector.
 */
Mat_<double> sparseMul(SparseMat_<double> A, Mat_<double> b);

/**
 * Checks that a given sparse matrix is symmetric.
 */
bool symmetric(Eigen::SparseMatrix<double> M);

/**
 * Checks that a given sparse matrix is positive definite by attempting
 * to compute its Cholesky decomposition.
 */
bool positiveDefinite(Eigen::SparseMatrix<double> M);

/**
 * Removes the line and column of a specific index in a sparse matrix.
 */
void removeLineCol(const Eigen::SparseMatrix<double> &L, int v0, Eigen::SparseMatrix<double> &L0);

/**
 * Converts 2d coordinates to upper triangular packed storage column major format
 * intended for use with LAPACK.
 *
 * @param rows the number of rows in the matrix
 * @param i the row index in the matrix
 * @param j the column index in the matrix
 * @return the linear index of the element in upper triangular packed storage column major format.
 */
int toUpperTriangularPacked(int i, int j);

/**
* Draws an histogram. Taken from
* http://laconsigna.wordpress.com/2011/04/29/1d-histogram-on-opencv/
*
* @param hist histogram to draw
* @param scaleX horizontal scaling factor
* @param scaleY vertical scaling factor
* @return histogram image
*/
Mat imHist(Mat hist, float scaleX=1, float scaleY=1);

/**
 * Computes and displays histograms for each channel of a color image.
 *
 * @param image the image to compute histograms from.
 */
void showHistograms(const Mat_<Vec3b> &image, const Mat_<float> &mask, int nbBins);
