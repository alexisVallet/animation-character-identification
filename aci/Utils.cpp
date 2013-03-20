#include "Utils.hpp"

int toRowMajor(int width, int x, int y) {
  return x + width * y;
}

pair<int,int> fromRowMajor(int width, int i) {
  pair<int,int> coords(i/width, i%width);

  return coords;
}
