#include "DisjointSet.hpp"

DisjointSetForest::DisjointSetForest() {

}

DisjointSetForest::DisjointSetForest(int numberOfElements) 
  : forest(numberOfElements), 
    numberOfComponents(numberOfElements),
    componentSizes(numberOfElements, 1),
    isModified(true)
{
  for (int i = 0; i < numberOfElements; i++) {
    this->forest[i].parent = i;
    this->forest[i].rank = 0;
  }
}

int DisjointSetForest::find(int element) {
  int currentParent = this->forest[element].parent;

  if (currentParent != element) {
    this->forest[element].parent = this->find(currentParent);
  }

  return this->forest[element].parent;
}

int DisjointSetForest::setUnion(int element1, int element2) {
  int root1 = this->find(element1);
  int root2 = this->find(element2);

  if (root1 == root2) {
    return root1;
  }

  // if the roots are different, then the union results in one
  // less component. We also indicate that the partition has been
  // modified so we must recompute root indexes.
  this->numberOfComponents--;
  this->isModified = true;

  if (this->forest[root1].rank < this->forest[root2].rank) {
    this->forest[root1].parent = root2;
    this->componentSizes[root2] += this->componentSizes[root1];
    return root2;
  } else if (this->forest[root1].rank > this->forest[root2].rank) {
    this->forest[root2].parent = root1;
    this->componentSizes[root1] += this->componentSizes[root2];
    return root1;
  } else {
    this->forest[root2].parent = root1;
    this->forest[root1].rank++;
    this->componentSizes[root1] += this->componentSizes[root2];
    return root1;
  }

}

Mat_<Vec<uchar, 3> > DisjointSetForest::toRegionImage(Mat_<Vec<uchar,3> > sourceImage) {
  Mat_<Vec<uchar, 3> > regions(sourceImage.rows, sourceImage.cols, CV_8UC3);
  vector<Vec<uchar,3> > colors(sourceImage.rows * sourceImage.cols);

  for (int i = 0; i < sourceImage.rows * sourceImage.cols; i++) {
    colors[i][0] = rand() % 255;
    colors[i][1] = rand() % 255;
    colors[i][2] = rand() % 255;
  }

  for (int i = 0; i < sourceImage.rows; i++) {
    for (int j = 0; j < sourceImage.cols; j++) {
      int root = this->find(toRowMajor(sourceImage.cols, j, i));

      regions(i, j) = colors[root];
    }
  }

  return regions;
}

int DisjointSetForest::getNumberOfComponents() {
  return this->numberOfComponents;
}

map<int,int> DisjointSetForest::getRootIndexes() {
  if (!this->isModified) {
    return this->rootIndexes;
  }
  // if the forest has been modified, recompute the indexes.
  this->rootIndexes.clear();
  int currentIndex = 0;

  for (int i = 0; i < (int)this->forest.size(); i++) {
    int root = this->find(i);
    map<int,int>::iterator it = this->rootIndexes.find(root);

    if (it == this->rootIndexes.end()) {
      this->rootIndexes[root] = currentIndex;
      currentIndex++;
    }
  }

  this->isModified = false;

  return this->rootIndexes;
}

int DisjointSetForest::getComponentSize(int element) {
  int root = this->find(element);

  return this->componentSizes[root];
}
