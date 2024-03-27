#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main() {

    Mat result, image;

    // high pass filter
    double filter[3][3]{
        {-1, -1, -1},
        {-1, +8, -1},
        {-1, -1, -1}
    };

    image = imread("tibetfox.bmp", IMREAD_GRAYSCALE);
    image.copyTo(result);
    // result.create(cv::Size(image.rows, image.cols), CV_8UC1);

    if (image.empty()) {
        cout << "image load failed!" << endl;
        return -1;
    }

    // Initialization
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            result.at<uchar>(y, x) = 0;
        }
    }

    // high pass filtering
    for (int y = 1; y < image.rows - 1; y++) {
        for (int x = 1; x < image.cols - 1; x++) {
            double res = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;
                    res += image.at<uchar>(ny, nx) * filter[ky + 1][kx + 1];
                }
            }      
            result.at<uchar>(y, x) = saturate_cast<uchar>(res);
        }
    }

    imshow("Original image", image);
    imshow("Result image", result);

    imwrite("tibetfox_hpf.bmp", result);
    waitKey(0);

    return 0;
}