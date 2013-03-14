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

template < typename T >
static float treeWalkKernelHelper(
    float (*basisKernel)(const T &l1,const T &l2), 
    int depth, 
    int arity, 
    LabelledGraph<T> &graph1,
    LabelledGraph<T> &graph2,
    graph_t &bGraph1,
    graph_t &bGraph2,
    embedding_t &embedding1,
    embedding_t &embedding2,
    int v1, 
    int v2,
    vector<vector<vector<float> > > &memoizedResults) {
  if (memoizedResults[depth-1][v1][v2] >= 0) {
    cout<<"returning memoized result"<<endl;

    return memoizedResults[depth-1][v1][v2];
  }
  if (depth == 1) {
    memoizedResults[depth-1][v1][v2] = basisKernel(graph1.getLabel(v1), graph2.getLabel(v2));
    
    return memoizedResults[depth-1][v1][v2];
  } else {
    // neighbors of v1 and v2 respectively in the circular order given
    // by the embeddings.
    vector<int> neighbors1;
    vector<int> neighbors2;
    property_traits<embedding_t>::value_type::const_iterator it;
    //    cout<<"extracting neighbors in circular order"<<endl;
    for (it = embedding1[v1].begin(); it != embedding1[v1].end(); it++) {
      neighbors1.push_back(get(vertex_index, bGraph1, target(*it, bGraph1)));
    }

    for (it = embedding2[v2].begin(); it != embedding2[v2].end(); it++) {
      neighbors2.push_back(get(vertex_index, bGraph2, target(*it, bGraph2)));
    }

    float sum = 0;
    
    for (int iSize = 1; iSize <= arity; iSize++) {
      //cout<<"for neighbor intervals of size "<<iSize<<endl;
      // enumerates all neighbor intervals of size iSize for both vertices.
      // If there are less neighbors than the required size, there are
      // no such intervals. Otherwise, there are exactly as many such
      // intervals as neighbors.
      if (neighbors1.size() >= iSize && neighbors2.size() >= iSize) {
	//cout<<"for each neighbor interval of "<<v1<<endl;
	// for each neighbor interval i of v1
	for (int i = 0; i < neighbors1.size(); i++) {
	  //cout<<"for each neighbor interval of "<<v2<<endl;
	  // for each neighbor interval j of v2
	  for (int j = 0; j < neighbors2.size(); j++) {
	    float product = 1;
	    
	    // for each neighbor r of v1 in i
	    for (int r = i; r < (i + iSize) % neighbors1.size(); r++) {
	      // for each neighbor s of v2 in j
	      for (int s = j; s < (j + iSize) % neighbors2.size(); s++) {
		//cout<<"computing for neighbors "<<r<<" and "<<s<<endl;
		product *= treeWalkKernelHelper(
		  basisKernel,
		  depth - 1,
		  arity,
		  graph1,
		  graph2,
		  bGraph1,
		  bGraph2,
		  embedding1,
		  embedding2,
		  r,
		  s,
		  memoizedResults);
	      }
	    }

	    sum += product;
	  }
	}
      }
    }

    memoizedResults[depth-1][v1][v2] = basisKernel(graph1.getLabel(v1), graph2.getLabel(v2)) * sum;
    
    return memoizedResults[depth-1][v1][v2];
  }
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
  // memoizes results for efficient dynamic programming recursion
  vector<vector<vector<float> > > memoizedResults(depth, vector<vector<float> >(graph1.numberOfVertices(), vector<float>(graph1.numberOfVertices(), -1)));

  float sum = 0;
  cout<<"computing boost graphs"<<endl;
  graph_t 
    bGraph1 = graph1.toBoostGraph(),
    bGraph2 = graph2.toBoostGraph();
  // data structures for boost
  embedding_storage_t 
    embedding_storage1(num_vertices(bGraph1)),
    embedding_storage2(num_vertices(bGraph2));
  embedding_t
    embedding1(embedding_storage1.begin(), get(vertex_index,bGraph1)),
    embedding2(embedding_storage2.begin(), get(vertex_index,bGraph2));
  cout<<"computing planar embeddings"<<endl;
  // computes planar embeddings for both graphs to get a circular
  // ordering of edges out of each vertex.
  bool isPlanar1 = boyer_myrvold_planarity_test(
     boyer_myrvold_params::graph = bGraph1,
     boyer_myrvold_params::embedding = embedding1);
  bool isPlanar2 = boyer_myrvold_planarity_test(
     boyer_myrvold_params::graph = bGraph2,
     boyer_myrvold_params::embedding = embedding2);
  assert(isPlanar1 && isPlanar2);

  for (int v1 = 0; v1 < graph1.numberOfVertices(); v1++) {
    for (int v2 = 0; v2 < graph2.numberOfVertices(); v2++) {
      cout<<"computing kernel for roots "<<v1<<" and "<<v2<<endl;
      sum += treeWalkKernelHelper(
	  basisKernel, 
	  depth, 
	  arity, 
	  graph1,
	  graph2,
	  bGraph1,
	  bGraph2,
	  embedding1, 
	  embedding2, 
	  v1, 
	  v2,
	  memoizedResults);
    }
  }  

  return sum;
}

#endif
