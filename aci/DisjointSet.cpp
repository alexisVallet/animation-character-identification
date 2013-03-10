 #include "DisjointSet.hpp"

DisjointSetForest::DisjointSetForest() {

}

DisjointSetForest::DisjointSetForest(int numberOfElements) 
  : forest(numberOfElements)
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

  if (this->forest[root1].rank < this->forest[root2].rank) {
    this->forest[root1].parent = root2;
    return root2;
  } else if (this->forest[root1].rank > this->forest[root2].rank) {
    this->forest[root2].parent = root1;
    return root1;
  } else {
    this->forest[root2].parent = root1;
    this->forest[root1].rank++;
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
