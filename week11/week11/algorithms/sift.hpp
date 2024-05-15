/*
 * 과제10: SIFT (Scale-Invariant Feature Transform)
 */

#pragma once
#include "opencv2\opencv.hpp"
#include <vector>

namespace algorithms
{
	struct DescriptorVector
	{
		int keypoint_id;
		float descriptor[8];
	};

	struct OctaveSet
	{
		int id;
		cv::Mat gaussianMat[6];
		cv::Mat gaussianDifferenceMat[5];
	};

	// p.258 기술자가 포함된 SIFT Keypoint P_i
	// (y_i, x_i, sigma_i, theta_i, octave_i
	struct KeyPoint
	{
		KeyPoint(int y, int x, int oct, double sig) : y(y), x(x), octave(oct), sigma(sig) {}
		int y;
		int x;
		double sigma;
		float dominantOrientation;
		int octave;
	};

	class SIFT
	{
	public:
		double octave_sigma_values[6];
		int octave_index_count;

		SIFT() : octave_index_count(0) {
			// Initialize octave_sigma_values
			octave_sigma_values[0] = 1.6;
			for (int i = 1; i < 6; ++i)
			{
				octave_sigma_values[i] = octave_sigma_values[i - 1] * 1.26;
			}
		}

		OctaveSet GenerateOctaveSet(const cv::InputArray& original_image);
		std::vector<KeyPoint> FindKeyPoints(const OctaveSet& octave_set);
		void DrawKeyPoints(const std::vector<KeyPoint>& key_points, const cv::InputArray& original_image, const cv::OutputArray& drawn_image);
		void FindDominantOrientation(std::vector<KeyPoint>& key_points, std::vector<OctaveSet>& octave_sets);
		std::vector<DescriptorVector> MakeDescriptor(std::vector<KeyPoint>& key_points, std::vector<OctaveSet>& octave_sets);
		int GetOctaveIdx(const double& sigma);
	};
}
