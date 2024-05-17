/*
 * ����10: SIFT Feature Distance
 * -> "algorithms/sift.hpp" ����
 */

#include "opencv2/opencv.hpp"
#include "algorithms/sift.hpp"
#include <chrono>
#include <cstdlib>
#include <ctime>

namespace algorithms
{
	// firstImgPtr���� ù��° �̹����� ��ǥ��,
	// secondImgPtr���� �ι�° �̹����� ��ǥ�� ���� �ȴ�.
	// �� �̹������� Euclidean Distance�� distance�� ��´�.
	struct PointSet
	{
		cv::Point firstImgPtr;
		cv::Point secondImgPtr;
		double distance;
	};
}

using namespace algorithms;

// �� ����Ʈ�� �Ÿ������� �������� �������ִ� std::sort�� ���� �Լ��̴�.
bool sort_by_distance_ascending(PointSet& a, PointSet& b)
{
	return b.distance > a.distance;
}

// ��ǥ ������ �Ÿ��� �����ϰ� ����Ѵ�.
// �� ��ǥ�� y, x ��ǥ ��� loosenPx���� ������ �پ��ִٸ�
// �ش� ��ǥ�� ������ ��ǥ�� �Ǵ��ϰ� true�� ��ȯ�Ѵ�.
bool compare_ptr_loosen(const cv::Point& a, const cv::Point& b, const double loosenPx = 2)
{
	if (abs(a.x - b.x) < loosenPx && abs(a.y - b.y) < loosenPx)
		return true;
	return false;
}

int main()
{
	// ������ �÷��� ���� �ʱ�ȭ �ڵ�
	std::srand(static_cast<unsigned int>(std::time(0)));

	cv::Mat first_image, second_image;
	cv::Mat first_image_gaussian, second_image_gaussian;
	cv::Mat first_image_output, second_image_output;

	first_image = cv::imread("images/florence-1.bmp", cv::IMREAD_GRAYSCALE);
	second_image = cv::imread("images/florence-2.bmp", cv::IMREAD_GRAYSCALE);

	cv::cvtColor(first_image, first_image_output, cv::COLOR_GRAY2BGR);
	cv::cvtColor(second_image, second_image_output, cv::COLOR_GRAY2BGR);

	assert(first_image.rows > 200 && first_image.cols > 200);
	assert(second_image.rows > 200 && second_image.cols > 200);

	cv::GaussianBlur(first_image, first_image_gaussian, cv::Size(0, 0), 1.6, 1.6);
	cv::GaussianBlur(second_image, second_image_gaussian, cv::Size(0, 0), 1.6, 1.6);

	/**
	 * 1�ܰ� - �ظ���-���ö� Ư¡(Ű����Ʈ) ���� �ܰ� (���� p.187 �˰��� 4-3)
	 * ����� Ű����Ʈ�� ��� image_keypoints�� ����ȴ�.
	 *
	 * first_image, second_image �� �̹��� ���ÿ� �����ϸ�,
	 * �� �̹����� OctaveSet�� KeyPoint�� _first, _second�� ������ ������ ����ȴ�.
	 */
	SIFT sift_first, sift_second;
	std::vector<OctaveSet> octave_sets_first, octave_sets_second;
	std::vector<KeyPoint> keypoints_first, keypoints_second;

	OctaveSet first_image_octave_set, second_image_octave_set;

	first_image_octave_set = sift_first.GenerateOctaveSet(first_image_gaussian);
	octave_sets_first.push_back(first_image_octave_set);
	second_image_octave_set = sift_second.GenerateOctaveSet(second_image_gaussian);
	octave_sets_second.push_back(second_image_octave_set);

	// ù��° �̹����� ���� KeyPoint ���� �� �߰����� OctaveSet�� �����Ѵ�.
	{
		std::vector<KeyPoint> octave_keypoint = sift_first.FindKeyPoints(first_image_octave_set);
		// image_keypoints = [image_keypoints] + [octave_keypoint]  (vector concatenation)
		keypoints_first.insert(keypoints_first.end(), octave_keypoint.begin(), octave_keypoint.end());

		for (int i = 0; i < 3; ++i)
		{
			// �������� �߰��� OctaveSet�� �����´�.
			OctaveSet octave_set = octave_sets_first.back();

			cv::Mat centerGaussianMat = octave_set.gaussianMat[3];
			cv::Mat halfSizeCenterGaussianMat;
			cv::resize(centerGaussianMat, halfSizeCenterGaussianMat, cv::Size(centerGaussianMat.cols / 2, centerGaussianMat.rows / 2));
			OctaveSet halfimg_octave_set = sift_first.GenerateOctaveSet(halfSizeCenterGaussianMat);
			octave_sets_first.push_back(halfimg_octave_set);

			std::vector<KeyPoint> halfimg_octave_keypoint = sift_first.FindKeyPoints(halfimg_octave_set);
			// image_keypoints = [image_keypoints] + [halfimg_octave_keypoint]  (vector concatenation)
			keypoints_first.insert(keypoints_first.end(), halfimg_octave_keypoint.begin(), halfimg_octave_keypoint.end());
		}
	}

	// �ι�° �̹����� ���� KeyPoint ���� �� �߰����� OctaveSet�� �����Ѵ�.
	{
		std::vector<KeyPoint> octave_keypoint = sift_second.FindKeyPoints(second_image_octave_set);
		// image_keypoints = [image_keypoints] + [octave_keypoint]  (vector concatenation)
		keypoints_second.insert(keypoints_second.end(), octave_keypoint.begin(), octave_keypoint.end());

		for (int i = 0; i < 3; ++i)
		{
			// �������� �߰��� OctaveSet�� �����´�.
			OctaveSet octave_set = octave_sets_second.back();

			cv::Mat centerGaussianMat = octave_set.gaussianMat[3];
			cv::Mat halfSizeCenterGaussianMat;
			cv::resize(centerGaussianMat, halfSizeCenterGaussianMat, cv::Size(centerGaussianMat.cols / 2, centerGaussianMat.rows / 2));
			OctaveSet halfimg_octave_set = sift_second.GenerateOctaveSet(halfSizeCenterGaussianMat);
			octave_sets_second.push_back(halfimg_octave_set);

			std::vector<KeyPoint> halfimg_octave_keypoint = sift_second.FindKeyPoints(halfimg_octave_set);
			// image_keypoints = [image_keypoints] + [halfimg_octave_keypoint]  (vector concatenation)
			keypoints_second.insert(keypoints_second.end(), halfimg_octave_keypoint.begin(), halfimg_octave_keypoint.end());
		}
	}

	/**
	 * 2�ܰ� - SIFT ����� ���� �ܰ� (���� p.258 �˰��� 6-1)
	 *
	 * ����, findDominantOrientation������ ����� Ű����Ʈ�� ���� �����Ͽ�
	 * �������� ���� theta_i�� ���ϰ� dominantOrientation property�� �����Ѵ�.
	 *
	 * �� ����, makeDescriptor���� Ư¡ ���� x_i�� DescriptorVector�� ������
	 * �� ����Ʈ�� ��ȯ�Ѵ�.
	 */

	sift_first.FindDominantOrientation(keypoints_first, octave_sets_first);
	sift_second.FindDominantOrientation(keypoints_second, octave_sets_second);

	std::vector<DescriptorVector> descriptor_vectors_first = sift_first.MakeDescriptor(keypoints_first, octave_sets_first);
	std::vector<DescriptorVector> descriptor_vectors_second = sift_second.MakeDescriptor(keypoints_second, octave_sets_second);

	std::cout << "1. ù��° �̹������� ����� ����� ����: " << descriptor_vectors_first.size() << "��" << std::endl;
	std::cout << "2. �ι�° �̹������� ����� ����� ����: " << descriptor_vectors_second.size() << "��" << std::endl;

	/**
	 * 3�ܰ� - SIFT Ư¡ ���͵� ���� Euclidean Distance ��� �ܰ�
	 * 
	 * �� �̹����κ��� ���� ����� Ư¡ ���͸� ��ȸ�ϸ鼭,
	 * ������ ���Ͱ��� Euclidean Distance�� ����Ѵ�.
	 * 
	 * �� ��, ���ϴ� ����� "ù��° �̹����� Ư¡ ����"�� 
	 * "�ι�° �̹����� Ư¡ ����"�̴�.
	 */
	std::vector<PointSet> distance_pair;

	for (int key_i = 0; key_i < keypoints_first.size(); key_i++) {
		for (int key_j = 0; key_j < keypoints_second.size(); key_j++) {

			// Ű����Ʈ���� ù��° ��Ÿ���� �̹��� Feature�� �̿��Ѵ�.
			// Ű����Ʈ �� ������ 16���� Ư¡���Ͱ� �������� Ȱ���Ͽ���.
			// desc_i or desc_j = 0, 16, 32, 48, ...
			for (int desc_i = 0; desc_i < descriptor_vectors_first.size(); desc_i += 16)
				for (int desc_j = 0; desc_j < descriptor_vectors_second.size(); desc_j += 16)
				{
					// Ư¡ ���� ����Ʈ�߿���, �츮�� ã������ Ű����Ʈ
					// key_i, key_j�� �ش��ϴ� Ư¡ ���͵��� ���ϵ��� �Ѵ�.
					if (desc_i / 16 != key_i || desc_j / 16 != key_j)
						continue;

					// Euclidean Distance (L2 Distance)�� ���Ѵ�.
					// �ǽ� PPT 3�������� ������ �����Ͽ� �����ϵ��� �Ѵ�.
					// ** ���ݺ��� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **

					double distance = 0.0;

					for (int k = 0; k < 8; ++k) {
						distance += pow(descriptor_vectors_first[desc_i].descriptor[k] - descriptor_vectors_second[desc_j].descriptor[k], 2);
					}

					distance = sqrt(distance);

					// ** ������� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **

					// Ư¡���� 2���� ���Ͽ� ���� Euclidean Distance��
					// 1.0���� ���� ��쿡�� distance_list�� �߰��ϵ��� �Ѵ�.
					// ��� ���� �ְ�, ���Ŀ� �� distance_list�� distance������
					// �������� �����Ͽ� ����غ����� �Ѵ�.
					if (distance < 1.0) {

						// ���� Ű����Ʈ�� Octave�� ������ �ش��ϴ� Scale�� ���Ѵ�.
						// ** ���ݺ��� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **
						
						int scale_first = pow(2, keypoints_first[key_i].octave);
						int scale_second = pow(2, keypoints_second[key_j].octave);
						
						
						// ** ������� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **

						// ���� Keypoint ����Ʈ���� key_i(ù��° �̹���), key_j(�ι�° �̹���)
						// �� �ε����� �ش��ϴ� Ű����Ʈ�κ��� ��ǥ�� ���Ͽ� �̸� firstImgPtr�� secondImgPtr�� �����Ѵ�.
						cv::Point firstImgPtr = cv::Point(keypoints_first[key_i].x, keypoints_first[key_i].y) * scale_first;
						cv::Point secondImgPtr = cv::Point(keypoints_second[key_j].x, keypoints_second[key_j].y) * scale_second;

						// ����Ʈ�� ���ų� ����� ���� �̹� �ִٸ�,
						// �Ÿ��� �� ª�� ������ �����ϰ� �� ���� ����Ʈ�� �߰����� �ʴ´�.
						bool ptrAlreadyInDistanceList = false;
						for (auto& point_set : distance_pair)
							if (compare_ptr_loosen(point_set.firstImgPtr, firstImgPtr)
								&& compare_ptr_loosen(point_set.secondImgPtr, secondImgPtr))
							{
								point_set.distance = MIN(point_set.distance, distance);
								ptrAlreadyInDistanceList = true;
								break;
							}

						if (!ptrAlreadyInDistanceList)
							distance_pair.push_back(PointSet{ firstImgPtr, secondImgPtr, distance });
					}
				}
		}
	}

	/**
	 * 4�ܰ� - Euclidean Distance ���� �� ȭ�� ǥ�� �ܰ�
	 *
	 * ���� Euclidean Distance���� distance ������ �������� �����Ѵ�.
	 * ��������� ����Ʈ�� ù��°�� �Ÿ��� ���� ����� ���� �ڸ���� �ȴ�.
	 * 
	 * ��� �̹����� ���� 5���� �� ���� ����Ѵ�.
	 */
	#//std::sort(distance_pair.begin(), distance_pair.end());
	// Sort by distance (lower distance fi	rst!)
	std::sort(distance_pair.begin(), distance_pair.end(), sort_by_distance_ascending);

	// Distance�� ª�� ������,
	// ���� 5���� ���� �׷��� ����Ѵ�.
	for (int i = 0; i < distance_pair.size() && i < 5; i++)
	{
		PointSet& point_set = distance_pair[i];

		std::cout << "PointSet #" + std::to_string(i + 1) << "\tDistance: " << point_set.distance << std::endl;
		std::cout << "\tFirst Image Point " << point_set.firstImgPtr.x << ", " << point_set.firstImgPtr.y << " (x, y)" << std::endl;
		std::cout << "\tSecond Image Point " << point_set.secondImgPtr.x << ", " << point_set.secondImgPtr.y << " (x, y)" << std::endl;


		cv::Scalar color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
		cv::circle(first_image_output, point_set.firstImgPtr, 5, color, 2);
		cv::circle(second_image_output, point_set.secondImgPtr, 5, color, 2);

	}

	// ȭ�鿡 ����� ǥ���Ѵ�.
	cv::imshow("key Point1", first_image_output);
	cv::imshow("key Point2", second_image_output);
	cv::waitKey();
	
	// �̹����� ���Ϸ� �����Ѵ�.
	cv::imwrite("florence-1-fv.bmp", first_image_output);
	cv::imwrite("florence-2-fv.bmp", second_image_output);
	std::cout << "���� ���� �Ϸ�!" << std::endl;
}
