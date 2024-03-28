/*
 * 과제4: Rotation using Homogeneous Matrix
 */

#include "opencv2/opencv.hpp"
#include "geometry.hpp"

int main() {
	cv::Mat original_image, forward_result_image, backward_result_image;
	original_image = cv::imread("cat.bmp", cv::IMREAD_GRAYSCALE);

	GeometryTransformator transformator;

	// 회전 동차 행렬을 정의한다.
	// geometry.cpp의 Geometry::SetRotateMatrix 함수를 정의하세요.
	double degree = -10.0;
	transformator.SetRotateMatrix(degree);

	// Q1. 전방 기하 변환을 정의한다.
	// geometry.cpp의 Geometry::ForwardTransformation 함수를 정의하세요.
	transformator.ForwardTransformation(original_image, forward_result_image);

	// Q2. 후방 기하 변환을 정의한다.
	// geometry.cpp의 Geometry::BackwardTransformation 함수를 정의하세요.
	transformator.BackwardTransformation(original_image, backward_result_image);

	cv::imshow("Original Image", original_image);
	cv::imshow("Forward Result Image", forward_result_image);
	cv::imshow("Backward Result Image", backward_result_image);
	cv::waitKey(0); // 아무 키나 누르면 계속한다.

	// 이미지를 파일로 저장한다.
	cv::imwrite("cat_forward.bmp", forward_result_image);
	cv::imwrite("cat_backward.bmp", backward_result_image);
	std::cout << "파일 저장 완료!" << std::endl;

	return 0;
}
