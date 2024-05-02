/*
 * 과제6: Moravec
 * -> "transformations/moravec.hpp" 참조
 */

#include "transformations/harriscorner.hpp"
#include "opencv2/opencv.hpp"
void main() {
	/// Load image
	cv::Mat bucks = imread("bucks.jpg", cv::IMREAD_GRAYSCALE);
	
	transformations::HarrisCorner harryscorner(bucks);
	harryscorner.FindConfidenceMap(bucks);

	cv::Mat result_image;

	cv::cvtColor(bucks, result_image, cv::COLOR_GRAY2BGR);
	harryscorner.DrawFeature(result_image);

	cv::imshow("Original Image", bucks);
	cv::imshow("Result Image", result_image);
	cv::imwrite("bucks_moravec.bmp", result_image);

	cv::waitKey();

}
