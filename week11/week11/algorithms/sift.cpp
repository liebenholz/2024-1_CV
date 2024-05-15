/*
 * ����10: SIFT (Scale-Invariant Feature Transform)
 */

#include "sift.hpp"
#include <string>

using namespace algorithms;

//
// ��Ÿ�� ��Ʈ�� ����� ��ȯ�Ѵ�.
// OctaveSet�� 8���� Gaussian Image�� DoG�� �̷���� �ϳ��� ��Ʈ�̴�. (���� p.189 ����)
// 
OctaveSet SIFT::GenerateOctaveSet(const cv::InputArray& octave_original_image)
{
	cv::Mat input_mat = octave_original_image.getMat();

	// ù��° ������ �������� ����Ǿ��ٰ� ������
	OctaveSet octave_set;
	memset(&octave_set, 0, sizeof(OctaveSet));

	octave_set.id = octave_index_count++;
	input_mat.copyTo(octave_set.gaussianMat[0]);

	// ù��° �̿��� �̹����� �ش��ϴ� sigma�� ��ŭ ����þ� ���� �� octave_set.gaussianMat �迭�� ����
	// (�Ʒ� API�� �̿��Ͽ� �ڵ带 �ۼ��մϴ�.)
	/*
		void GaussianBlur(InputArray src, OutputArray dst, Size ksize, double sigmaX);
		@src ���� ����(input_mat)
		@dst ��� ����(octave_set.gaussianMat �迭)
		@ksize Ŀ�� ������(cv::Size(0, 0))
		@sigmaX sigma(octave_sigma_values �迭)
	*/
	for (int i = 1; i < 6; ++i)
	{
		// ** ���ݺ��� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **
		cv::GaussianBlur(input_mat, octave_set.gaussianMat[i], cv::Size(0, 0), octave_sigma_values[i]);
		// ** ������� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **
	}

	// Gaussian �̹����� �̿��Ͽ� DoG (Difference of Gaussian)�� ���Ѵ�.
	for (int i = 1; i < 6; ++i)
	{
		// octave_set.gaussianMat ����þ� �̹����� �̿��Ͽ�,
		// octave_set.gaussianDifferenceMat�� DoG�� �����ϼ���.
		// ** ���ݺ��� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **
		octave_set.gaussianDifferenceMat[i - 1] = octave_set.gaussianMat[i] - octave_set.gaussianMat[i - 1]; // octave_set.gaussianMat[i - 1] - octave_set.gaussianMat[i]; // octave_set.gaussianMat[i] - octave_set.gaussianMat[i - 1]
		// ** ������� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **
	}

	// ������� ��Ÿ�� ��Ʈ�� ��ȯ�Ѵ�.
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
				// Ư¡���� ���� ���ΰ�(Threshold)�� �����Ͽ�
				// �ʹ� ���� ���� ����
				if (octave_set.gaussianDifferenceMat[octave_idx].at<uchar>(h, w) <= 11)  // ����10: 8, ����11: 11
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
				//�ִ�, �ּڰ��� ���� ��ǥ�̸� Ű����Ʈ(Ư¡��)���� �Ǵ�
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
	// ���� ������׷� ����
	const int SIFT_ORI_HIST_BINS = 36;
	for (int i = 0; i < key_points.size(); ++i)
	{
		KeyPoint& key_point = key_points[i];

		/*
			���� p.256
			Hess�� ������ ��������� 1.5 x 3 x Sigma�� �ݿø��Ͽ� w�� ���� ��
			(2w+1) x (2w+1)�� �����츦 ����Ͽ���.
		*/
		int radius = 1.5 * 3 * key_point.sigma;

		std::vector<double> angles;	
		std::vector<double> magnitudes;

		// Octave�� Sigma�� �´� �������� ó��
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
			// bin = 10�� �������� ����ȭ�� ���� ���� ��
			// cvRound �Լ��� SIFT_ORI_HIST_BINS ����� �̿��Ͽ� bin���� �ݿø��Ͽ� ���ϼ���.
			// ** ���ݺ��� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **
			int bin = cvRound((SIFT_ORI_HIST_BINS / 360.f) * angles[s]);
			// ** ������� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **

			if (bin >= SIFT_ORI_HIST_BINS)
				bin -= SIFT_ORI_HIST_BINS;
			if (bin < 0)
				bin += SIFT_ORI_HIST_BINS;
			// ������׷��� ���� ������ ����� ������ ������ ���
			hist[bin] += (1 * magnitudes[s]);
		}

		float maxval = hist[0];
		float maxOri = 0;
		// ���� �ִ��� ������׷� ���� Dominant Orienation���� ����
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

	// ���� ������׷� ����
	const int SIFT_ORI_HIST_BINS = 8;
	for (int i = 0; i < key_points.size(); ++i)
	{
		KeyPoint& key = key_points[i];
		/*
			���� p.257
			Hess�� ������ ��������� siz�� 3 x sigma�� �����Ͽ���.
		*/
		int siz = 3 * key.sigma;

		// 36���� ����ȭ�Ͽ� 10���� ���� ������׷��� ������ ������
		// ���� ������ ����ϱ� ���ؼ� 10�� ������
		float orientation = key.dominantOrientation * 10;

		cv::Point beginPoint = cv::Point(key.x - (2 * siz), key.y - (2 * siz));
		cv::Point endPoint = cv::Point(key.x + (2 * siz), key.y + (2 * siz));

		// Octave�� Sigma�� �´� �������� ó��
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

						// ���� ���� ��� ���
						cv::Mat homogeneousResult = orgIdx.t() * m.t();
						int w2_rotated = homogeneousResult.at<double>(0);
						int h2_rotated = homogeneousResult.at<double>(1);

						// ���� ��ġ�� Angle�� Magnitude ����
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
					// bin = 8�ܰ�� ����ȭ�� ���� ���� ��.
					// cvRound �Լ��� SIFT_ORI_HIST_BINS ����� �̿��Ͽ� bin���� �ݿø��Ͽ� ���ϼ���.
					// ** ���ݺ��� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **
					int bin = cvRound((SIFT_ORI_HIST_BINS / 360.f) * angles[s]);
					// ** ������� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **

					if (bin >= SIFT_ORI_HIST_BINS)
						bin -= SIFT_ORI_HIST_BINS;
					if (bin < 0)
						bin += SIFT_ORI_HIST_BINS;

					vector.descriptor[bin] += (1 * magnitudes[s]);
				}

				float sum_of_histogram = 0;

				// ����ȭ ����
				// Descriptor ���鿡 ���� ����ȭ�� �����մϴ�.
				// SIFT_ORI_HIST_BINS���� ������׷� �� vector.descriptor�鿡 ����,
				// ��� ���� ���� �� sum_of_histogram���� �� element�� �������ݴϴ�.
				
				// ** ���ݺ��� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **
				for (int quant_radius = 0; quant_radius < SIFT_ORI_HIST_BINS; ++quant_radius)
					sum_of_histogram += vector.descriptor[quant_radius];

				for (int quant_radius = 0; quant_radius < SIFT_ORI_HIST_BINS; ++quant_radius)
					if (sum_of_histogram != 0)
						vector.descriptor[quant_radius] /= sum_of_histogram;
				// ** ������� �ڵ带 �ۼ��ϼ���. �� ���� ����ø� �� �˴ϴ� **

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
