/*
 * ����4: Rotation using Homogeneous Matrix
 */

#include "geometry.hpp"
#include "opencv2\opencv.hpp"

void GeometryTransformator::ForwardTransformation(cv::InputArray& srcMat, cv::OutputArray& dstMat) {
	if (!hasMatrixSet)
		throw "���� ����� ���ǵ��� �ʾҽ��ϴ�. ���� ����� ���� �������ּ���.";

	cv::Mat input_image, output_image;

	// ���ڷκ��� Mat �̹��� �ε�
	input_image = srcMat.getMat();
	if (dstMat.empty())
		dstMat.create(input_image.size(), input_image.type());
	output_image = dstMat.getMat();

	/**
	 * input_image�� �̹����� Forward Tranformation�ϴ� �ڵ带 �����ϼ���.
	 * input_image�� H ������ �̿��Ͽ� Transformation�ϰ�, ����� output_image�� �����ϼ���.
	 */
	 // ** ���ݺ��� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **

	for (int y = 0; y < input_image.rows; y++) {
		for (int x = 0; x < input_image.cols; x++) {

			double transformed_y = H[0][0] * y + H[1][0] * x + H[2][0];
			double transformed_x = H[0][1] * y + H[1][1] * x + H[2][1];

			int y_prime = static_cast<int>(transformed_y);
			int x_prime = static_cast<int>(transformed_x);

			if (y_prime < input_image.rows && x_prime < input_image.cols && y_prime >= 0 && x_prime >= 0) {
				output_image.at<uchar>(y_prime, x_prime) = input_image.at<uchar>(y, x);
			}
		}
	}

	// ** ������� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **

}

void GeometryTransformator::BackwardTransformation(cv::InputArray& srcMat, cv::OutputArray& dstMat) {
	if (!hasMatrixSet)
		throw "���� ����� ���ǵ��� �ʾҽ��ϴ�. ���� ����� ���� �������ּ���.";

	cv::Mat input_image, output_image;

	// ���ڷκ��� Mat �̹��� �ε�
	input_image = srcMat.getMat();
	if (dstMat.empty())
		dstMat.create(input_image.size(), input_image.type());
	output_image = dstMat.getMat();

	double H_inverse[3][3] = {};
	InverseHmatrix(H_inverse);

	/**
	 * input_image�� �̹����� Backward Tranformation�ϴ� �ڵ带 �����ϼ���.
	 * input_image�� H_inverse ������ �̿��Ͽ�
	 * GeometryTransformator::ForwardTransformation�� �����ϰ� �����ϸ� �˴ϴ�.
	 */
	 // ** ���ݺ��� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **

	for (int y = 0; y < input_image.rows; y++) {
		for (int x = 0; x < input_image.cols; x++) {

			double transformed_y = H_inverse[0][0] * y + H_inverse[1][0] * x + H_inverse[2][0];
			double transformed_x = H_inverse[0][1] * y + H_inverse[1][1] * x + H_inverse[2][1];

			int y_prime = static_cast<int>(transformed_y);
			int x_prime = static_cast<int>(transformed_x);

			// �̹��� ũ�� ���� ���� �ִ� ���鸸 �����Ѵ�.
			if (y_prime < input_image.rows && x_prime < input_image.cols && y_prime >= 0 && x_prime >= 0) {
				output_image.at<uchar>(y, x) = input_image.at<uchar>(y_prime, x_prime);
			}
		}
	}

	// ** ������� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **
}

void GeometryTransformator::SetMoveMatrix(double y, double x) {
	ClearMatrix();

	H[0][0] = 1;
	H[1][1] = 1;
	H[2][0] = y;
	H[2][1] = x;
	H[2][2] = 1;

	hasMatrixSet = true;
}

void GeometryTransformator::SetRotateMatrix(double degree) {
	ClearMatrix();

	/**
	 * 3*3 ��� ���� H�� �̵� ���� ����� �ۼ��ϼ���.
	 * ������ ���ڸ��� ���� �Ҵ��ϸ� �˴ϴ�.
	 * cos() �Լ��� sin() �Լ��� ���ڴ� degree�� �ƴ� radian�� �Է¹޴� ���� �����ϰ�
	 * PI ������ Ȱ���Ͽ� ��ȯ�մϴ�.
	 */

	 // ** ���ݺ��� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **

	double radian = degree * PI / 180.0;
	H[0][0] = cos(radian);
	H[0][1] = -sin(radian);
	H[1][0] = sin(radian);
	H[1][1] = cos(radian);
	H[2][2] = 1;

	// ** ������� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **

	hasMatrixSet = true;
}

void GeometryTransformator::InverseHmatrix(double result[3][3]) {
	double determinant = 0;

	for (int i = 0; i < 3; i++)
		determinant += (H[0][i] * (H[1][(i + 1) % 3] * H[2][(i + 2) % 3] - H[1][(i + 2) % 3] * H[2][(i + 1) % 3]));

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			result[i][j] = (
				(H[(j + 1) % 3][(i + 1) % 3] * H[(j + 2) % 3][(i + 2) % 3]) -
				(H[(j + 1) % 3][(i + 2) % 3] * H[(j + 2) % 3][(i + 1) % 3])
				) / determinant;
		}
	}
}

void GeometryTransformator::ClearMatrix() {
	memset(H, 0, sizeof(H));
	hasMatrixSet = false;
}
