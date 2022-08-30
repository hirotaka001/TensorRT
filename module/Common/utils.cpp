// Martin Kersner, m.kersner@gmail.com
// 2016/12/19 

#include <kros/trt/Common/utils.hpp>
using std::vector;

cv::Rect VecToRect(const vector<float>& vec)
{
	return cv::Rect(cv::Point((int)vec[0], (int)vec[1]), cv::Point((int)vec[2], (int)vec[3]));
}

void DrawRectangles(cv::Mat& img,
	const vector<vector<float>>& vecVecFloat)
{
	for (const auto& vec : vecVecFloat)
		cv::rectangle(img, VecToRect(vec), WHITE_COLOR);
}

void DrawRectangles(cv::Mat& img,
	const vector<cv::Rect>& vecRect)
{
	for (const auto& rect : vecRect)
		cv::rectangle(img, rect, WHITE_COLOR);
}
