#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <utility>

using namespace std;

/**
 * Converts coordinates in 2D array to row major format.
 */
int toRowMajor(int width, int x, int y);
/**
 * Converts a row major index to coordinates in a 2D array.
 */
pair<int,int> fromRowMajor(int width, int i);

#endif
