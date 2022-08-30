// Martin Kersner, m.kersner@gmail.com
// 2016/12/18

#ifndef NMS_HPP__
#define NMS_HPP__

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>

#ifdef _WIN32
#pragma warning(pop)
#endif

enum PointInRectangle { XMIN, YMIN, XMAX, YMAX, CLASS, SCORE };

std::vector<std::vector<float> > nms(const std::vector<std::vector<float>>& boxes,
	const float& threshold,
	const bool sort_input = true,
	const int overlap_method = 0);

std::vector<float> GetPointFromRect(const std::vector<std::vector<float>>&,
	const PointInRectangle&);

std::vector<float> ComputeArea(const std::vector<float>&,
	const std::vector<float>&,
	const std::vector<float>&,
	const std::vector<float>&);

template <typename T>
std::vector<int> argsort(const std::vector<T>& v);

std::vector<float> Maximum(const float&,
	const std::vector<float>&);

std::vector<float> Minimum(const float&,
	const std::vector<float>&);

std::vector<float> CopyByIndexes(const std::vector<float>&,
	const std::vector<int>&);

std::vector<int> RemoveLast(const std::vector<int>&);

std::vector<float> Subtract(const std::vector<float>&,
	const std::vector<float>&);

std::vector<float> Multiply(const std::vector<float>&,
	const std::vector<float>&);

// vec( A∩B / A )
std::vector<float> Divide_AandB_by_A(
	const std::vector<float>&,// vec A∩B
	const std::vector<float>&);// A

// vec( A∩B / A∪B )
std::vector<float> Divide_AandB_by_AorB(
	const std::vector<float>&,// vec A∩B
	const std::vector<float>&,// vec A
	const float&);// B

// vec( A∩B / B )
std::vector<float> Divide_AandB_by_B(
	const std::vector<float>&,// vec A∩B
	const float&);// B

std::vector<int> WhereLarger(const std::vector<float>&,
	const float&);

std::vector<int> RemoveByIndexes(const std::vector<int>&,
	const std::vector<int>&);

std::vector<std::vector<float> >  BoxesToRectangles(const std::vector<std::vector<float>>&);

template <typename T>
std::vector<T> FilterVector(const std::vector<T>&,
	const std::vector<int>&);

#endif // NMS_HPP__
