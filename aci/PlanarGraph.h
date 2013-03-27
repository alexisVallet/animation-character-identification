#pragma once
#include "LabeledGraph.hpp"
#include <opencv2\opencv.hpp>

template < typename T >
class PlanarGraph :	public LabeledGraph<T>
{
private:
	vector<Point_<float> > vertexPositions;

public:
	PlanarGraph(void);
	~PlanarGraph(void);
};
