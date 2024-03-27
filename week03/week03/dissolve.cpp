#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void dissolve(Mat img1, Mat img2, Mat res, float a) {

    for (int y = 0; y < res.rows; y++) {
        for (int x = 0; x < res.cols; x++) {
            int b = a * img1.at<Vec3b>(y, x)[0] + (1 - a) * img2.at<Vec3b>(y, x)[0];
            int g = a * img1.at<Vec3b>(y, x)[1] + (1 - a) * img2.at<Vec3b>(y, x)[1];
            int r = a * img1.at<Vec3b>(y, x)[2] + (1 - a) * img2.at<Vec3b>(y, x)[2];

            res.at<Vec3b>(y, x)[0] = saturate_cast<uchar>(b);
            res.at<Vec3b>(y, x)[1] = saturate_cast<uchar>(g);
            res.at<Vec3b>(y, x)[2] = saturate_cast<uchar>(r);
        }
    }
}

int main() {

    Mat result, image1, image2;
    image1 = imread("cat.bmp", IMREAD_COLOR);
    image2 = imread("tibetfox.bmp", IMREAD_COLOR);

    if (image1.empty() || image2.empty()) {
        cout << "image load failed!" << endl;
        return -1;
    }

    if (image1.rows != image2.rows || image1.cols != image2.cols) {
        cout << "different image size!" << endl;
        return -1;
    }

    // result.create(cv::Size(image1.rows, image1.cols), CV_8UC1);
    image1.copyTo(result);

    // imshow("First image", image1);
    // imshow("Second image", image2);

    // a = 0.3
    dissolve(image1, image2, result, 0.3);
    imshow("Dissolved image 3", result);
    imwrite("dissolve_3.bmp", result);

    // a = 0.7
    dissolve(image1, image2, result, 0.7);
    imshow("Dissolved image 7", result);
    imwrite("dissolve_7.bmp", result);

    waitKey(0);
    return 0;
}