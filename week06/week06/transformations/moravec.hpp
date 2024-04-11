#pragma once
#include "opencv2\opencv.hpp"

namespace transformations
{
	class Moravec
	{
	private:
		cv::Mat confidence_map;

	public:
		Moravec(cv::InputArray& original_image);
		void FindConfidenceMap(cv::InputArray& original_image);
		void DrawFeature(cv::OutputArray& result_image);
	};
}