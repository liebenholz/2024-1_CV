
#include "opencv2/opencv.hpp"
#include "filter.hpp"

int main1() {
	cv::Mat original_image, sobelXOut, sobelYOut;
	original_image = cv::imread("tibetfox.bmp", cv::IMREAD_GRAYSCALE);

	imshow("original", original_image);

	// SobelFilterY�� CreateMask()�� �����ϼ���.
	SobelFilterY sobelYFilter;
	sobelYFilter.CreateMask();
	sobelYFilter.Convolute(original_image, sobelYOut);

	// SobelFilterX�� CreateMask()�� �����ϼ���.
	SobelFilterX sobelXFilter;
	sobelXFilter.CreateMask();
	sobelXFilter.Convolute(original_image, sobelXOut);

	cv::Mat edgeStrength;
	edgeStrength.create(original_image.size(), CV_8UC1);

	/**
	 * sobelYOut�� sobelXOut�� �̿��Ͽ� Edge Strength Map�� ���ϼ���.
	 */
	// ** ���ݺ��� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **
	for (int y = 0; y < original_image.rows; y++) {
		for (int x = 0; x < original_image.cols; x++) {
			double dy = sobelYOut.at<uchar>(y, x);
			double dx = sobelXOut.at<uchar>(y, x);
			double magnitude = sqrt(dy * dy + dx * dx);

			edgeStrength.at<uchar>(y, x) = (int)((magnitude < 255) ? magnitude : 255);
		}
	}
	// ** ������� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **

	cv::imshow("sobelY-Output", sobelYOut);
	cv::imshow("sobelX-Output", sobelXOut);
	cv::imshow("Edge Strength", edgeStrength);
	cv::waitKey();

	// �̹����� ���Ϸ� �����Ѵ�.
	cv::imwrite("tibetfox_y.bmp", sobelYOut);
	cv::imwrite("tibetfox_x.bmp", sobelXOut);
	cv::imwrite("tibetfox_strength.bmp", edgeStrength);
	std::cout << "���� ���� �Ϸ�!" << std::endl;

	return 0;
}