/*
 * 과제10: SIFT Feature Distance
 * -> "algorithms/sift.hpp" 참조
 */

#include "opencv2/opencv.hpp"
#include "algorithms/sift.hpp"
#include <chrono>
#include <cstdlib>
#include <ctime>

namespace algorithms
{
	// firstImgPtr에는 첫번째 이미지의 좌표가,
	// secondImgPtr에는 두번째 이미지의 좌표가 담기게 된다.
	// 두 이미지간의 Euclidean Distance를 distance에 담는다.
	struct PointSet
	{
		cv::Point firstImgPtr;
		cv::Point secondImgPtr;
		double distance;
	};
}

using namespace algorithms;

// 두 포인트를 거리순으로 오름차순 정렬해주는 std::sort의 헬퍼 함수이다.
bool sort_by_distance_ascending(PointSet& a, PointSet& b)
{
	return b.distance > a.distance;
}

// 좌표 사이의 거리를 느슨하게 계산한다.
// 두 좌표가 y, x 좌표 모두 loosenPx보다 가깝게 붙어있다면
// 해당 좌표는 동일한 좌표로 판단하고 true를 반환한다.
bool compare_ptr_loosen(const cv::Point& a, const cv::Point& b, const double loosenPx = 2)
{
	if (abs(a.x - b.x) < loosenPx && abs(a.y - b.y) < loosenPx)
		return true;
	return false;
}

int main()
{
	// 랜덤한 컬러를 위한 초기화 코드
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
	 * 1단계 - 해리스-라플라스 특징(키포인트) 검출 단계 (교재 p.187 알고리즘 4-3)
	 * 검출된 키포인트는 모두 image_keypoints에 저장된다.
	 *
	 * first_image, second_image 두 이미지 동시에 진행하며,
	 * 각 이미지의 OctaveSet과 KeyPoint는 _first, _second로 구별된 변수에 저장된다.
	 */
	SIFT sift_first, sift_second;
	std::vector<OctaveSet> octave_sets_first, octave_sets_second;
	std::vector<KeyPoint> keypoints_first, keypoints_second;

	OctaveSet first_image_octave_set, second_image_octave_set;

	first_image_octave_set = sift_first.GenerateOctaveSet(first_image_gaussian);
	octave_sets_first.push_back(first_image_octave_set);
	second_image_octave_set = sift_second.GenerateOctaveSet(second_image_gaussian);
	octave_sets_second.push_back(second_image_octave_set);

	// 첫번째 이미지에 대한 KeyPoint 검출 및 추가적인 OctaveSet을 생성한다.
	{
		std::vector<KeyPoint> octave_keypoint = sift_first.FindKeyPoints(first_image_octave_set);
		// image_keypoints = [image_keypoints] + [octave_keypoint]  (vector concatenation)
		keypoints_first.insert(keypoints_first.end(), octave_keypoint.begin(), octave_keypoint.end());

		for (int i = 0; i < 3; ++i)
		{
			// 마지막에 추가한 OctaveSet을 가져온다.
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

	// 두번째 이미지에 대한 KeyPoint 검출 및 추가적인 OctaveSet을 생성한다.
	{
		std::vector<KeyPoint> octave_keypoint = sift_second.FindKeyPoints(second_image_octave_set);
		// image_keypoints = [image_keypoints] + [octave_keypoint]  (vector concatenation)
		keypoints_second.insert(keypoints_second.end(), octave_keypoint.begin(), octave_keypoint.end());

		for (int i = 0; i < 3; ++i)
		{
			// 마지막에 추가한 OctaveSet을 가져온다.
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
	 * 2단계 - SIFT 기술자 추출 단계 (교재 p.258 알고리즘 6-1)
	 *
	 * 먼저, findDominantOrientation에서는 검출된 키포인트를 각각 접근하여
	 * 지배적인 방향 theta_i를 구하고 dominantOrientation property에 저장한다.
	 *
	 * 그 다음, makeDescriptor에서 특징 벡터 x_i를 DescriptorVector로 생성해
	 * 그 리스트를 반환한다.
	 */

	sift_first.FindDominantOrientation(keypoints_first, octave_sets_first);
	sift_second.FindDominantOrientation(keypoints_second, octave_sets_second);

	std::vector<DescriptorVector> descriptor_vectors_first = sift_first.MakeDescriptor(keypoints_first, octave_sets_first);
	std::vector<DescriptorVector> descriptor_vectors_second = sift_second.MakeDescriptor(keypoints_second, octave_sets_second);

	std::cout << "1. 첫번째 이미지에서 검출된 기술자 개수: " << descriptor_vectors_first.size() << "개" << std::endl;
	std::cout << "2. 두번째 이미지에서 검출된 기술자 개수: " << descriptor_vectors_second.size() << "개" << std::endl;

	/**
	 * 3단계 - SIFT 특징 벡터들 간의 Euclidean Distance 계산 단계
	 * 
	 * 두 이미지로부터 각각 계산한 특징 벡터를 순회하면서,
	 * 각각의 벡터간의 Euclidean Distance를 계산한다.
	 * 
	 * 이 때, 비교하는 대상은 "첫번째 이미지의 특징 벡터"와 
	 * "두번째 이미지의 특징 벡터"이다.
	 */
	std::vector<PointSet> distance_pair;

	for (int key_i = 0; key_i < keypoints_first.size(); key_i++) {
		for (int key_j = 0; key_j < keypoints_second.size(); key_j++) {

			// 키포인트마다 첫번째 옥타브의 이미지 Feature만 이용한다.
			// 키포인트 한 개마다 16개의 특징벡터가 생성됨을 활용하였다.
			// desc_i or desc_j = 0, 16, 32, 48, ...
			for (int desc_i = 0; desc_i < descriptor_vectors_first.size(); desc_i += 16)
				for (int desc_j = 0; desc_j < descriptor_vectors_second.size(); desc_j += 16)
				{
					// 특징 벡터 리스트중에서, 우리가 찾으려는 키포인트
					// key_i, key_j에 해당하는 특징 벡터들을 비교하도록 한다.
					if (desc_i / 16 != key_i || desc_j / 16 != key_j)
						continue;

					// Euclidean Distance (L2 Distance)를 구한다.
					// 실습 PPT 3페이지의 수식을 참고하여 구현하도록 한다.
					// ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **

					double distance = 0.0;

					for (int k = 0; k < 8; ++k) {
						distance += pow(descriptor_vectors_first[desc_i].descriptor[k] - descriptor_vectors_second[desc_j].descriptor[k], 2);
					}

					distance = sqrt(distance);

					// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **

					// 특징벡터 2개를 비교하여 구한 Euclidean Distance가
					// 1.0보다 작을 경우에만 distance_list에 추가하도록 한다.
					// 모든 값을 넣고, 추후에 이 distance_list를 distance순으로
					// 오름차순 정렬하여 출력해보도록 한다.
					if (distance < 1.0) {

						// 현재 키포인트의 Octave의 제곱에 해당하는 Scale을 구한다.
						// ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
						
						int scale_first = pow(2, keypoints_first[key_i].octave);
						int scale_second = pow(2, keypoints_second[key_j].octave);
						
						
						// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **

						// 구한 Keypoint 리스트에서 key_i(첫번째 이미지), key_j(두번째 이미지)
						// 각 인덱스에 해당하는 키포인트로부터 좌표를 구하여 이를 firstImgPtr와 secondImgPtr에 저장한다.
						cv::Point firstImgPtr = cv::Point(keypoints_first[key_i].x, keypoints_first[key_i].y) * scale_first;
						cv::Point secondImgPtr = cv::Point(keypoints_second[key_j].x, keypoints_second[key_j].y) * scale_second;

						// 리스트에 같거나 비슷한 쌍이 이미 있다면,
						// 거리가 더 짧은 값으로 설정하고 그 쌍은 리스트에 추가하지 않는다.
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
	 * 4단계 - Euclidean Distance 정렬 및 화면 표시 단계
	 *
	 * 계산된 Euclidean Distance들을 distance 순으로 오름차순 정렬한다.
	 * 결과적으로 리스트의 첫번째로 거리가 가장 가까운 쌍이 자리잡게 된다.
	 * 
	 * 결과 이미지에 상위 5개의 원 쌍을 출력한다.
	 */
	#//std::sort(distance_pair.begin(), distance_pair.end());
	// Sort by distance (lower distance fi	rst!)
	std::sort(distance_pair.begin(), distance_pair.end(), sort_by_distance_ascending);

	// Distance가 짧은 순으로,
	// 상위 5개만 원을 그려서 출력한다.
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

	// 화면에 결과를 표시한다.
	cv::imshow("key Point1", first_image_output);
	cv::imshow("key Point2", second_image_output);
	cv::waitKey();
	
	// 이미지를 파일로 저장한다.
	cv::imwrite("florence-1-fv.bmp", first_image_output);
	cv::imwrite("florence-2-fv.bmp", second_image_output);
	std::cout << "파일 저장 완료!" << std::endl;
}
