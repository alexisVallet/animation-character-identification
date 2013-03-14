#ifndef _TREEWALK_KERNEL_
#define _TREEWALK_KERNEL_

#include <opencv2/opencv.hpp>
#include <boost/graph/boyer_myrvold_planar_test.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/property_map/property_map.hpp>
#include <vector>
#include <iostream>
#include <cmath>

#include "LabelledGraph.hpp"
#include "DisjointSet.hpp"

using namespace std;
using namespace cv;
using namespace boost;

typedef adjacency_list<vecS,vecS,bidirectionalS,property<vertex_index_t, int> > graph_t;

typedef vector<vector<graph_traits<adjacency_list<vecS,vecS,bidirectionalS,property<vertex_index_t, int> > >::edge_descriptor> > embedding_storage_t;

typedef iterator_property_map<vector<vector<graph_traits<adjacency_list<vecS,vecS,bidirectionalS,property<vertex_index_t, int> > >::edge_descriptor> >::iterator,property_map<adjacency_list<vecS,vecS,bidirectionalS,property<vertex_index_t, int> >,vertex_index_t>::type> embedding_t;

/**
 * Adds color histogram labels to a segmentation graph.
 * 
 * @param image image to compute hitograms from
 * @param segmentation a segmentation of the image
 * @param segmentationGraph segmentation graph to add labels to
 * @param binsPerChannel the number of histogram bins per color channel
 */
void colorHistogramLabels(
  Mat_<Vec<uchar,3> > &image, 
  DisjointSetForest &segmentation, 
  LabelledGraph<Mat> &segmentationGraph,
  int binsPerChannel);

/**
 * Labels a segmentation graph by median color of each segment.
 *
 * @param image image to compute hitograms from
 * @param segmentation a segmentation of the image
 * @param segmentationGraph segmentation graph to add labels to
 */
void medianColorLabels(
  Mat_<Vec<uchar,3> > &image,
  DisjointSetForest &segmentation,
  LabelledGraph<Vec<uchar,3> > &segmentationGraph);

/**
 * khi² (X²) kernel between two color histograms seen as
 * distributions.
 */
float khi2Kernel(int binsPerChannel, float lambda, float mu, const Mat &h1, const Mat &h2);

float kroneckerKernel(const Mat &h1, const Mat &h2);

static int current(int iteration) {
  return iteration % 2;
}

static int previous(int iteration) {
  return !current(iteration);
}

/**
 * Computes the tree walk kernel between two labelled graphs given
 * a basis kernel function. It is assumed that each graph is undirected,
 * with each edge duplicated in both the source and destination's
 * adjacency list. Both graphs must be planar.
 *
 * @param basisKernel basis kernel function, taking 2 labels as parameters.
 * @param graph1 the first graph
 * @param graph2 the second graph
 * @return the tree walk kernel value between graph1 and graph2.
 */
template < typename T >
float treeWalkKernel(float (*basisKernel)(const T &l1,const T &l2), int depth, int arity, LabelledGraph<T> &graph1, LabelledGraph<T> &graph2) {
  // data structures for boost's boyer myrvold planarity test implementation
  graph_t 
    bGraph1 = graph1.toBoostGraph(),
    bGraph2 = graph2.toBoostGraph();
  embedding_storage_t 
    embedding_storage1(num_vertices(bGraph1)),
    embedding_storage2(num_vertices(bGraph2));
  embedding_t
    embedding1(embedding_storage1.begin(), get(vertex_index,bGraph1)),
    embedding2(embedding_storage2.begin(), get(vertex_index,bGraph2));

  // computes planar embeddings for both graphs to get a circular
  // ordering of edges out of each vertex.
  bool isPlanar1 = boyer_myrvold_planarity_test(
     boyer_myrvold_params::graph = bGraph1,
     boyer_myrvold_params::embedding = embedding1);
  bool isPlanar2 = boyer_myrvold_planarity_test(
     boyer_myrvold_params::graph = bGraph2,
     boyer_myrvold_params::embedding = embedding2);
  assert(isPlanar1 && isPlanar2);

  // basisKernels contains the basis kernel for each pair of vertices
  // the two graphs, computed once and for all.
  Mat_<float> basisKernels(graph1.numberOfVertices(), graph2.numberOfVertices(), CV_32F);

  // depthKernels contains the current iteration's results as well as the previous
  // iteration's results, swapped at each iteration in a "ping-pong" fashion.
  Mat_<float> depthKernels[2] = {
    Mat_<float>(graph1.numberOfVertices(), graph2.numberOfVertices(), CV_32F),
    Mat_<float>(graph1.numberOfVertices(), graph2.numberOfVertices(), CV_32F)
  };

  // neighbors for each vertex in each graph in the circular order given
  // by the embeddings.
  vector<vector<int> > 
    circNeighbors1(graph1.numberOfVertices()), 
    circNeighbors2(graph2.numberOfVertices());

  //initializes neighbors in circular order
  for (int v1 = 0; v1 < graph1.numberOfVertices(); v1++) {
    property_traits<embedding_t>::value_type::const_iterator it;

    for (it = embedding1[v1].begin(); it != embedding1[v1].end(); it++) {
      circNeighbors1[v1].push_back(get(vertex_index, bGraph1, target(*it, bGraph1)));
    }
  }
  for (int v2 = 0; v2 < graph2.numberOfVertices(); v2++) {
    property_traits<embedding_t>::value_type::const_iterator it;

    for (it = embedding2[v2].begin(); it != embedding2[v2].end(); it++) {
      circNeighbors2[v2].push_back(get(vertex_index, bGraph2, target(*it, bGraph2)));
    }
  }

  // initializes basis kernels
  for (int v1 = 0; v1 < graph1.numberOfVertices(); v1++) {
    for (int v2 = 0; v2 < graph2.numberOfVertices(); v2++) {
      basisKernels(v1,v2) = basisKernel(graph1.getLabel(v1), graph2.getLabel(v2));
      depthKernels[previous(0)](v1,v2) = basisKernels(v1,v2);
    }
  }

  // computes kernels for each depth up to the maximum depth
  for (int d = 0; d < depth; d++) {
    for (int v1 = 0; v1 < graph1.numberOfVertices(); v1++) {
      for (int v2 = 0; v2 < graph2.numberOfVertices(); v2++) {
	//cout<<"from roots "<<v1<<" and "<<v2<<endl;
	float sum = 0;
	
	// summing for neighbor intervals of cardinal lower
	// than arity
	for (int iSize = 1; iSize <= arity; iSize++) {
	  // if the size is greater than either number of
	  // neighbors, there are no such neighbor intervals
	  if (iSize > circNeighbors1[v1].size() || iSize > circNeighbors2[v2].size()) {
	    break;
	  }
	  // for each neighbor interval i (resp j) of v1 (resp v2)
	  for (int i = 0; i < circNeighbors1[v1].size() - iSize; i++) {
	    for (int j = 0; j < circNeighbors2[v2].size() - iSize; j++) {
	      float product = 1;
	      // for each neighbor r (resp s) of v1 (resp v2) in i (resp j)
	      for (int r = i; r < i + iSize; r++) {
		for (int s = j; s < j + iSize; s++) {
		  product *= depthKernels[previous(d)](r,s);
		}
	      }
	      sum += product;
	    }
	  }
	}

	depthKernels[current(d)](v1,v2) = basisKernels(v1,v2) * sum;
      }
    }
  }
  
  Scalar result = sum(depthKernels[previous(depth)]);

  // sum and return the results of the last iteration
  return result[0];
}

#endif
