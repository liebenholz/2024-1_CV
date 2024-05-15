/*
 * 과제10: SIFT (Scale-Invariant Feature Transform)
 */

#include "sift.hpp"
#include <string>

using namespace algorithms;

//
// 옥타브 세트를 만들어 반환한다.
// OctaveSet은 8개의 Gaussian Image와 DoG로 이루어진 하나의 세트이다. (교재 p.189 참조)
// 
OctaveSet SIFT::GenerateOctaveSet(const cv::InputArray& octave_original_image)
{
	cv::Mat input_mat = octave_original_image.getMat();

	// 첫번째 영상은 스무딩이 적용되었다고 가정함
	OctaveSet octave_set;
	memset(&octave_set, 0, sizeof(OctaveSet));

	octave_set.id = octave_index_count++;
	input_mat.copyTo(octave_set.gaussianMat[0]);

	// 첫번째 이외의 이미지에 해당하는 sigma값 만큼 가우시안 적용 후 octave_set.gaussianMat 배열에 저장
	// (아래 API를 이용하여 코드를 작성합니다.)
	/*
		void GaussianBlur(InputArray src, OutputArray dst, Size ksize, double sigmaX);
		@src 원본 영상(input_mat)
		@dst 결과 영상(octave_set.gaussianMat 배열)
		@ksize 커널 사이즈(cv::Size(0, 0))
		@sigmaX sigma(octave_sigma_values 배열)
	*/
	for (int i = 1; i < 6; ++i)
	{
		// ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
		cv::GaussianBlur(input_mat, octave_set.gaussianMat[i], cv::Size(0, 0), octave_sigma_values[i]);
		// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
	}

	// Gaussian 이미지를 이용하여 DoG (Difference of Gaussian)을 구한다.
	for (int i = 1; i < 6; ++i)
	{
		// octave_set.gaussianMat 가우시안 이미지를 이용하여,
		// octave_set.gaussianDifferenceMat에 DoG를 생성하세요.
		// ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
		octave_set.gaussianDifferenceMat[i - 1] = octave_set.gaussianMat[i] - octave_set.gaussianMat[i - 1]; // octave_set.gaussianMat[i - 1] - octave_set.gaussianMat[i]; // octave_set.gaussianMat[i] - octave_set.gaussianMat[i - 1]
		// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
	}

	// 만들어진 옥타브 세트를 반환한다.
	return octave_set;
}

//
std::vector<KeyPoint> SIFT::FindKeyPoints(const OctaveSet& octave_set)
{
	
	std::vector<KeyPoint> key_points;

	for (int octave_idx = 1; octave_idx <= 3; ++octave_idx)
		for (int h = 1; h < octave_set.gaussianDifferenceMat[octave_idx].rows - 1; ++h)
			for (int w = 1; w < octave_set.gaussianDifferenceMat[octave_idx].cols - 1; ++w)
			{
				// 특징점의 값에 문턱값(Threshold)을 적용하여
				// 너무 작은 값은 배제
				if (octave_set.gaussianDifferenceMat[octave_idx].at<uchar>(h, w) <= 11)  // 과제10: 8, 과제11: 11
					continue;

				int max_v = octave_set.gaussianDifferenceMat[octave_idx].at<uchar>(h, w);
				int min_v = octave_set.gaussianDifferenceMat[octave_idx].at<uchar>(h, w);

				for (int o2 = octave_idx - 1; o2 <= octave_idx + 1; ++o2)
					for (int h2 = h - 1; h2 <= h + 1; ++h2)
						for (int w2 = w - 1; w2 <= w + 1; ++w2)
						{
							if (octave_idx == o2 && h == h2 && w == w2)
								continue;

							if (octave_set.gaussianDifferenceMat[o2].at<uchar>(h2, w2) > max_v)
								max_v = octave_set.gaussianDifferenceMat[o2].at<uchar>(h2, w2);

							if (octave_set.gaussianDifferenceMat[o2].at<uchar>(h2, w2) < min_v)
								min_v = octave_set.gaussianDifferenceMat[o2].at<uchar>(h2, w2);
						}
				//최댓, 최솟값이 현재 좌표이면 키포인트(특징점)으로 판단
				if (max_v == octave_set.gaussianDifferenceMat[octave_idx].at<uchar>(h, w) || 
					min_v == octave_set.gaussianDifferenceMat[octave_idx].at<uchar>(h, w))
					key_points.push_back(KeyPoint( h, w, octave_set.id, octave_sigma_values[octave_idx]));
			}
	

	return key_points;
}

void SIFT::DrawKeyPoints(const std::vector<KeyPoint> &key_points, const cv::InputArray& original_image, const cv::OutputArray& drawn_image)
{
	if (original_image.channels() == 1)
		cv::cvtColor(original_image, drawn_image, cv::COLOR_GRAY2BGR);
	else
		original_image.copyTo(drawn_image);

	cv::Mat drawn_mat = drawn_image.getMat();
	for (int i = 0; i < key_points.size(); ++i)
	{
		int scale = pow(2, key_points[i].octave);
		cv::circle(drawn_mat, cv::Point(key_points[i].x * scale, key_points[i].y * scale), 3, cv::Scalar(255, 0, 0));
	}
}


void SIFT::FindDominantOrientation(std::vector<KeyPoint>& key_points, std::vector<OctaveSet>& octave_sets)
{
	// 에지 히스토그램 개수
	const int SIFT_ORI_HIST_BINS = 36;
	for (int i = 0; i < key_points.size(); ++i)
	{
		KeyPoint& key_point = key_points[i];

		/*
			교재 p.256
			Hess가 구현한 방법에서는 1.5 x 3 x Sigma를 반올림하여 w를 구한 후
			(2w+1) x (2w+1)의 윈도우를 사용하였다.
		*/
		int radius = 1.5 * 3 * key_point.sigma;

		std::vector<double> angles;	
		std::vector<double> magnitudes;

		// Octave와 Sigma에 맞는 영상으로 처리
		int sigmaIdx = GetOctaveIdx(key_point.sigma);
		assert((0 <= sigmaIdx) && (sigmaIdx < 6));

		cv::Mat img = octave_sets[key_point.octave].gaussianMat[sigmaIdx];

		cv::Mat img_sobel_x, img_sobel_y;
		cv::Mat magnitude_map, angle_map;

		cv::Sobel(img, img_sobel_x, CV_64FC1, 1, 0);
		cv::Sobel(img, img_sobel_y, CV_64FC1, 0, 1);

		cv::cartToPolar(img_sobel_x, img_sobel_y, magnitude_map, angle_map, true);

		for (int h = -radius; h <= radius; ++h)
		{
			for (int w = -radius; w <= radius; w++)
			{
				int y = key_point.y + h;
				int x = key_point.x + w;

				if (x <= 0 || x >= img.cols - 1 || y <= 0 || y >= img.rows - 1)
					continue;

				angles.push_back(angle_map.at<double>(y, x));
				magnitudes.push_back(magnitude_map.at<double>(y, x));
			}
		}

		float hist[SIFT_ORI_HIST_BINS] = { 0, };
		for (int s = 0; s < angles.size(); ++s) // also iterates magnitude
		{
			// bin = 10도 간격으로 양자화된 에지 방향 값
			// cvRound 함수와 SIFT_ORI_HIST_BINS 상수를 이용하여 bin값을 반올림하여 구하세요.
			// ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
			int bin = cvRound((SIFT_ORI_HIST_BINS / 360.f) * angles[s]);
			// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **

			if (bin >= SIFT_ORI_HIST_BINS)
				bin -= SIFT_ORI_HIST_BINS;
			if (bin < 0)
				bin += SIFT_ORI_HIST_BINS;
			// 히스토그램의 값을 에지의 방향과 강도의 곱으로 계산
			hist[bin] += (1 * magnitudes[s]);
		}

		float maxval = hist[0];
		float maxOri = 0;
		// 값이 최대인 히스토그램 값을 Dominant Orienation으로 설정
		for (int j = 1; j < SIFT_ORI_HIST_BINS; j++)
			if (maxval < hist[j])
			{
				maxval = hist[j];
				maxOri = j;
			}

		key_point.dominantOrientation = maxOri;
	}
}

std::vector<DescriptorVector> SIFT::MakeDescriptor(std::vector<KeyPoint>& key_points, std::vector<OctaveSet>& octave_sets)
{
	std::vector<DescriptorVector> descriptor_vectors;

	// 에지 히스토그램 개수
	const int SIFT_ORI_HIST_BINS = 8;
	for (int i = 0; i < key_points.size(); ++i)
	{
		KeyPoint& key = key_points[i];
		/*
			교재 p.257
			Hess가 구현한 방법에서는 siz를 3 x sigma로 설정하였다.
		*/
		int siz = 3 * key.sigma;

		// 36개로 양자화하여 10도씩 에지 히스토그램을 만들어기 때문에
		// 원래 에지를 계산하기 위해서 10을 곱해줌
		float orientation = key.dominantOrientation * 10;

		cv::Point beginPoint = cv::Point(key.x - (2 * siz), key.y - (2 * siz));
		cv::Point endPoint = cv::Point(key.x + (2 * siz), key.y + (2 * siz));

		// Octave와 Sigma에 맞는 영상으로 처리
		int sigmaIdx = GetOctaveIdx(key.sigma);
		const cv::Mat& img = octave_sets[key.octave].gaussianMat[sigmaIdx];

		cv::Mat img_sobel_x, img_sobel_y;
		cv::Mat magnitude_map, angle_map;

		cv::Sobel(img, img_sobel_x, CV_64FC1, 1, 0);
		cv::Sobel(img, img_sobel_y, CV_64FC1, 0, 1);

		cv::cartToPolar(img_sobel_x, img_sobel_y, magnitude_map, angle_map, true);
		cv::Mat m = cv::getRotationMatrix2D(cv::Point(key.x, key.y), orientation, 1);

		for (int h = beginPoint.y; h < endPoint.y; h += siz)
		{
			for (int w = beginPoint.x; w < endPoint.x; w += siz)
			{
				std::vector<double> angles;
				std::vector<double> magnitudes;
				for (int h2 = h; h2 < h + siz; h2++)
				{
					for (int w2 = w; w2 < w + siz; w2++)
					{
						cv::Mat orgIdx(cv::Size(1, 3), CV_64FC1);

						orgIdx.at<double>(0) = w2;
						orgIdx.at<double>(1) = h2;
						orgIdx.at<double>(2) = 1.0;

						// 복합 동차 행렬 계산
						cv::Mat homogeneousResult = orgIdx.t() * m.t();
						int w2_rotated = homogeneousResult.at<double>(0);
						int h2_rotated = homogeneousResult.at<double>(1);

						// 계산된 위치의 Angle과 Magnitude 저장
						if (w2_rotated > 0 && w2_rotated < img.cols - 1 && h2_rotated > 0 && h2_rotated < img.rows - 1)
						{
							angles.push_back(angle_map.at<double>(h2_rotated, w2_rotated));
							magnitudes.push_back(magnitude_map.at<double>(h2_rotated, w2_rotated));
						}
					}
				}

				DescriptorVector vector;
				memset(&vector, 0, sizeof(DescriptorVector));
				vector.keypoint_id = i;

				for (int s = 0; s < angles.size(); ++s)
				{
					// bin = 8단계로 양자화된 에지 방향 값.
					// cvRound 함수와 SIFT_ORI_HIST_BINS 상수를 이용하여 bin값을 반올림하여 구하세요.
					// ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
					int bin = cvRound((SIFT_ORI_HIST_BINS / 360.f) * angles[s]);
					// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **

					if (bin >= SIFT_ORI_HIST_BINS)
						bin -= SIFT_ORI_HIST_BINS;
					if (bin < 0)
						bin += SIFT_ORI_HIST_BINS;

					vector.descriptor[bin] += (1 * magnitudes[s]);
				}

				float sum_of_histogram = 0;

				// 정규화 구현
				// Descriptor 값들에 대한 정규화를 진행합니다.
				// SIFT_ORI_HIST_BINS개의 히스토그램 값 vector.descriptor들에 대해,
				// 모든 값을 합한 값 sum_of_histogram으로 각 element를 나누어줍니다.
				
				// ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
				for (int quant_radius = 0; quant_radius < SIFT_ORI_HIST_BINS; ++quant_radius)
					sum_of_histogram += vector.descriptor[quant_radius];

				for (int quant_radius = 0; quant_radius < SIFT_ORI_HIST_BINS; ++quant_radius)
					if (sum_of_histogram != 0)
						vector.descriptor[quant_radius] /= sum_of_histogram;
				// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **

				descriptor_vectors.push_back(vector);
			}
		}
	}

	return descriptor_vectors;
}

int SIFT::GetOctaveIdx(const double &sigma)
{
	double epsilon = 1e-6;

	for (int i = 0; i < 6; ++i)
		if (sigma == octave_sigma_values[i]) // roughly sigma == sig[i]
			return i;

	// Failed to find appropriate sigma value!
	assert(false);
	return -1;
}
