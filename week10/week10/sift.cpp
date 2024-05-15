#include "SIFT.h"
#include <string>
void SIFT_c::InitSig() {
	sig[0] = 1.6;
	for(int i = 1; i < 6; ++i) {
		sig[i] = sig[i-1] * 1.26;
	}
}


void SIFT_c::MakeOctaveImg(cv::Mat& org) {
	// ù��° ������ �������� ����Ǿ��ٰ� ������
	Octave oct;
	oct.octave = octaveVec.size();
	org.copyTo(oct.gauImage[0]);

	// �̿��� �̹����� �ش��ϴ� �ñ׸� ��ŭ ����þ� ����
	/*
	CV_EXPORTS_W void GaussianBlur( InputArray src, OutputArray dst, Size ksize,
                                double sigmaX, double sigmaY = 0,
                                int borderType = BORDER_DEFAULT );
	src - ���� ����(org)
	dst - ��� ����(oct.gauImage array)
	ksize - Ŀ�� ������
	sigmaX - sigma(sig array)
	*/
	for(int i = 1; i < 6; ++i) {
		// ****** ���� 1 ******	
		// ù��° �̹��� �̿��� �̹����� �ش��ϴ� �ñ׸� ��ŭ ����þ� �����Ͻÿ�.
		// ����ǥ(???)�� ������ �ڵ带 �Է��Ͻÿ�.
		cv::GaussianBlur(org, oct.gauImage[i], cv::Size(), sig[i]);
	}

	octaveVec.push_back(oct);
}

void SIFT_c::DOG() {
	Octave &oct = octaveVec[octaveVec.size()-1];
	for(int i = 1; i < 6; ++i) {		
		// ****** ���� 2 ******	
		// oct.gauImage array�� �̿��Ͽ� oct.DoG array�� DOG�� �����Ͻÿ�.
		// ����ǥ(???)�� ������ �ڵ带 �Է��Ͻÿ�.
		oct.DoG[i-1] = oct.gauImage[i-1] - oct.gauImage[i];
	}
}

void SIFT_c::FindKeyPoint()
{	
	int octIdx = octaveVec.size()-1;
	Octave &oct = octaveVec[octIdx];

	for(int o = 1; o <= 3; ++o) {		
		for(int h = 1; h < oct.DoG[o].rows - 1; ++h) {
			for(int w = 1; w < oct.DoG[o].cols - 1; ++w) {
				// Ư¡���� ���� ���ΰ��� �����Ͽ�
				// �ʹ� ���� ���� ����
				if(oct.DoG[o].at<uchar>(h,w) <= 8)
					continue;
				int max_v = oct.DoG[o].at<uchar>(h,w);
				int min_v = oct.DoG[o].at<uchar>(h,w);
				
				for(int o2 = o-1; o2 <= o+1; ++o2) {
					for(int h2 = h - 1; h2 <= h + 1; ++h2) {
						for(int w2 = w - 1; w2 <= w + 1; ++w2) {
							if(o == o2 && h == h2 && w == w2)
								continue;							

							if(oct.DoG[o2].at<uchar>(h2,w2) > max_v) {
								max_v = oct.DoG[o2].at<uchar>(h2,w2);
							}

							if(oct.DoG[o2].at<uchar>(h2,w2) < min_v) {
								min_v = oct.DoG[o2].at<uchar>(h2,w2);
							}
						}
					}				
				}
				//�ִ�, �ּڰ��� ���� ��ǥ�̸� Ű����Ʈ(Ư¡��)���� �Ǵ�
				if(max_v == oct.DoG[o].at<uchar>(h,w))						
					keyPointVec.push_back(keyPoint(h, w, octIdx, sig[o]));
				if(min_v == oct.DoG[o].at<uchar>(h,w))						
					keyPointVec.push_back(keyPoint(h, w, octIdx, sig[o]));				
			}
		}
	}		
}

void SIFT_c::DrawKeyPoint(cv::Mat &dst) {
	for(int i = 0; i < keyPointVec.size(); ++i) {
		int scale = pow(2, keyPointVec[i].octave);
		circle(dst, cv::Point(keyPointVec[i].x * scale,keyPointVec[i].y)*scale, 3, cv::Scalar(255, 0, 0));
	}
}
