#include <algorithm>
#include "harriscorner.hpp"

transformations::HarrisCorner::HarrisCorner(cv::InputArray& original_image)
{
	cv::Mat src = original_image.getMat();
	confidence_map.create(src.size(), CV_32FC1);

	// 외곽 2개 픽셀의 Confidence는 0으로 가정
	for (int w = 0; w < src.cols; ++w)
	{
		confidence_map.at<float>(0, w) = 0.0;
		confidence_map.at<float>(1, w) = 0.0;
		confidence_map.at<float>(src.rows - 1, w) = 0.0;
		confidence_map.at<float>(src.rows - 2, w) = 0.0;
	}

	for (int h = 0; h < src.rows; ++h)
	{
		confidence_map.at<float>(h, 0) = 0.0;
		confidence_map.at<float>(h, 1) = 0.0;
		confidence_map.at<float>(h, src.cols - 1) = 0.0;
		confidence_map.at<float>(h, src.cols - 2) = 0.0;
	}
}


void transformations::HarrisCorner::FindConfidenceMap(cv::InputArray& original_image)
{
	cv::Mat src = original_image.getMat();

	/**
	 * src를 이용하여 해리스코너의  알고리즘의 Confidence를 계산한 뒤
	 * 이를 confidence_map에 저장하세요.
	 * (confidence_map은 클래스 멤버 변수로 선언되어 있습니다.)
	 * 가우시안 필터는 클래스의 멤버 변수 G로 선언되어 있습니다. 크기는 3x3입니다.
	 * 가장자리는 0으로 하시면 됩니다.
	 * k값은 0.04로 하시면 됩니다.
	 */
	 // ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **

	// 마스크 선언
	cv::Mat dx, dy;
	cv::Mat dyy, dxx, dyx;
	cv::Mat Gdyy, Gdxx, Gdyx;

	// Sobel Filter
	int sobelY[3][3] = {
		{-1, -2, -1},
		{0, 0, 0},
		{1, 2, 1}
	};

	int sobelX[3][3] = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1}
	};
	double k = 0.04;

	// 마스크 생성
	dy.create(src.size(), CV_32FC1);
	dx.create(src.size(), CV_32FC1);
	dyy.create(src.size(), CV_32FC1);
	dxx.create(src.size(), CV_32FC1);
	dyx.create(src.size(), CV_32FC1);
	Gdyy.create(src.size(), CV_32FC1);
	Gdxx.create(src.size(), CV_32FC1);
	Gdyx.create(src.size(), CV_32FC1);

	// 마스크 초기화(가장자리 포함)
	for (int h = 0; h < src.rows; ++h) {
		for (int w = 0; w < src.cols; ++w) {
			dy.at<float>(h, w) = 0.0;
			dx.at<float>(h, w) = 0.0;
			dyy.at<float>(h, w) = 0.0;
			dxx.at<float>(h, w) = 0.0;
			dyx.at<float>(h, w) = 0.0;
		}
	}

	// 1. Sobel Filter를 이용한 dy, dx 연산
	for (int y = 1; y < src.rows - 1; y++) {	
		for (int x = 1; x < src.cols - 1; x++) {	
			double sumY = 0.0, sumX = 0.0;

			for (int j = 0; j < 3; j++) {
				for (int i = 0; i < 3; i++) {
					sumY += src.at<uchar>(y + j - 1, x + i - 1) * sobelY[j][i];
					sumX += src.at<uchar>(y + j - 1, x + i - 1) * sobelX[j][i];
				}
			}
			dy.at<float>(y, x) = sumY;
			dx.at<float>(y, x) = sumX;
		}
	}

	// dyy, dxx ,dyx 연산
	for (int y = 1; y < src.rows - 1; y++) {	
		for (int x = 1; x < src.cols - 1; x++) {							
			dyy.at<float>(y, x) = dy.at<float>(y, x) * dy.at<float>(y, x);
			dxx.at<float>(y, x) = dx.at<float>(y, x) * dx.at<float>(y, x);
			dyx.at<float>(y, x) = dx.at<float>(y, x) * dy.at<float>(y, x);
		}
	}
	
	// 2. Gaussian Filter 적용
	for (int y = 2; y < src.rows - 2; y++) {
		for (int x = 2; x < src.cols - 2; x++) {
			float sumXX = 0.0, sumYY = 0.0, sumYX = 0.0;

			for (int j = 0; j < 3; j++) {
				for (int i = 0; i < 3; i++) {
					sumYY += dyy.at<float>(y + j - 1, x + i - 1) * G[j][i];
					sumXX += dxx.at<float>(y + j - 1, x + i - 1) * G[j][i];
					sumYX += dyx.at<float>(y + j - 1, x + i - 1) * G[j][i];
				}
			}
			Gdyy.at<float>(y, x) = sumYY;
			Gdxx.at<float>(y, x) = sumXX;
			Gdyx.at<float>(y, x) = sumYX;
		}
	}

	// 3. Confidence 연산 후 confidence_map에 값 저장
	for (int y = 2; y < src.rows - 2; y++) {
		for (int x = 2; x < src.cols - 2; x++) {
			float p = Gdyy.at<float>(y, x);
			float q = Gdxx.at<float>(y, x);
			float r = Gdyx.at<float>(y, x);

			confidence_map.at<float>(y, x) = (p * q - r * r) - (k * pow(p + q, 2));
		}
	}
	// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
}


void transformations::HarrisCorner::DrawFeature(cv::OutputArray& result_image)
{
	cv::Mat dst = result_image.getMat();
	cv::Mat confidence_map_norm ;
	cv::normalize(confidence_map, confidence_map_norm, 0, 1, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

	for (int h = 0; h < confidence_map_norm.rows; ++h) {
		for (int w = 0; w < confidence_map_norm.cols; ++w) {
			if (confidence_map_norm.at<float>(h, w) >= 0.3) {
				cv::circle(dst, cv::Point(w, h), 3, cv::Scalar(255, 0, 0));
			}
		}
	}
}
