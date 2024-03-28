/*
 * °úÁ¦4: Rotation using Homogeneous Matrix
 */

#pragma once
#include "opencv2\opencv.hpp"

class GeometryTransformator {
private:
	const double PI;
	double H[3][3] = {};
	bool hasMatrixSet;

public:
	GeometryTransformator() : PI(acos(-1)), hasMatrixSet(false) {}

	void ForwardTransformation(cv::InputArray& srcMat, cv::OutputArray& dstMat);
	void BackwardTransformation(cv::InputArray& srcMat, cv::OutputArray& dstMat);

	void SetMoveMatrix(double y, double x);
	void SetRotateMatrix(double degree);

	void InverseHmatrix(double result[3][3]);
	void ClearMatrix();
};