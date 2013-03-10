/**
 * Implementation of a disjoint set forest data structure with rank
 * and path compression.
 */
#ifndef _DISJOINTSET_HPP_
#define _DISJOINTSET_HPP_

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "Utils.hpp"

using namespace std;
using namespace cv;

struct DisjointSet {
  int parent;
  int rank;
};

class DisjointSetForest {
private:
  vector<DisjointSet> forest;

public:
  DisjointSetForest(); // should not be called
  /**
   * Initializes the forest with numberOfElements singleton disjoint
   * sets, numbered 0 to numberOfElements-1.
   *
   * @param numberOfElements the number of elements in the partition,
   * corresponding to the number of singletons at initialization.
   */
  DisjointSetForest(int numberOfElements);
  /**
   * Returns the representant of the set containing a specific element.
   *
   * @param element an element from a set.
   * @return the representant of the set containing the element.
   */
  int find(int element);
  /**
   * Unifies two sets in the forest into one, represented by an element
   * from each set, returning the new root (or representant) of the set.
   *
   * @param element1 an element from the first set.
   * @param element2 an element from the second set.
   * @return the new root (or representant) of the set.
   */
  int setUnion(int element1, int element2);
  /**
   * Computes a region image assuming each element in the set forest
   * is a pixel in row major order, labelled by their set representant.
   */
  Mat_<Vec<uchar,3> > toRegionImage(Mat_<Vec<uchar,3> > sourceImage);
};

#endif
