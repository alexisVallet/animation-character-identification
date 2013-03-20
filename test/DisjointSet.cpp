#include <stdio.h>
#include <iostream>

#include "DisjointSet.hpp"

#define FOREST_SIZE 100
#define UNION_NUMBER 50

using namespace std;

int main() {
  DisjointSetForest testForest(FOREST_SIZE);
  vector<int> randomSources(UNION_NUMBER);
  vector<int> randomDestinations(UNION_NUMBER);

  // Initial forest only has singletons
  for (int i = 0; i < FOREST_SIZE; i++) {
    assert(testForest.find(i) == i);
  }

  for (int i = 0; i < UNION_NUMBER; i++) {
    randomSources[i] = rand() % FOREST_SIZE;
    randomDestinations[i] = rand() % FOREST_SIZE;

    testForest.setUnion(randomSources[i], randomDestinations[i]);
  }

  // Checks that union'd elements have the same root
  for (int i = 0; i < UNION_NUMBER; i++) {
    assert(testForest.find(randomSources[i]) 
	   == testForest.find(randomDestinations[i]));
  }

  return 0;
}
