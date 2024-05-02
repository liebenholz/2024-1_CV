#pragma once
#include "opencv2\opencv.hpp"

namespace transformations
{
	class HarrisCorner
	{
	private:
		cv::Mat confidence_map;
		double G[3][3] = {{ 0.0751, 0.1238, 0.0751},
			{ 0.1238, 0.2042, 0.1238},
			{0.0751,0.1238,0.0751}};

	public:
		HarrisCorner(cv::InputArray& original_image);
		void FindConfidenceMap(cv::InputArray& original_image);
		void DrawFeature(cv::OutputArray& result_image);
	};
}