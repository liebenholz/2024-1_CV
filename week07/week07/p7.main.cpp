/*
 * 과제7: LOG (Laplacian of Gaussian)
 * -> "transformations/filter.hpp" 참조
 */

#include "opencv2/opencv.hpp"
#include "filter.hpp"

int main()
{
	cv::Mat original_image, convoluted_image, zerocrossing_image;
	original_image = cv::imread("bucks.jpg", cv::IMREAD_GRAYSCALE);

	double sigma = 2.0; // 0.5, 1.0, 2.0, ...
	LOG logFilter;

	// CreateMask()를 구현하세요.
	logFilter.CreateMask(sigma);
	logFilter.Convolute(original_image, convoluted_image);
	// FindZeroCrossing()과 CheckSign()을 구현하세요.
	logFilter.FindZeroCrossing(convoluted_image, zerocrossing_image);

	cv::imshow("Original Image", original_image);
	cv::imshow("Convoluted Image", convoluted_image);
	cv::imshow("Zero-Crossing Image", zerocrossing_image);
	cv::waitKey();

	// 이미지를 파일로 저장한다.
	cv::imwrite("bucks_conv.bmp", convoluted_image);
	cv::imwrite("bucks_zerocrossing.bmp", zerocrossing_image);
	std::cout << "파일 저장 완료!" << std::endl;
}