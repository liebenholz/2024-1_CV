#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main() {

    Mat result, image;

    // low pass filter
    double filter[3][3]{
        {1 / 9.0, 1 / 9.0, 1 / 9.0},
        {1 / 9.0, 1 / 9.0, 1 / 9.0},
        {1 / 9.0, 1 / 9.0, 1 / 9.0}
    };

    // color
    image = imread("salt_pepper.bmp", IMREAD_COLOR);
    image.copyTo(result);

    if (image.empty()) {
        cout << "image load failed!" << endl;
        return -1;
    }

    // Initialization
    // Put original pixel amount at the edge of the result image.
    for (int y = 1; y < image.rows - 1; y++) {
        for (int x = 1; x < image.cols - 1; x++) {
            result.at<uchar>(y, x) = 0;
        }
    }

    // low pass filtering
    for (int y = 1; y < image.rows - 1; y++) {
        for (int x = 1; x < image.cols - 1; x++) {
            double b = 0, g = 0, r = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;
                    b += image.at<Vec3b>(ny, nx)[0] * filter[ky + 1][kx + 1];
                    g += image.at<Vec3b>(ny, nx)[1] * filter[ky + 1][kx + 1];
                    r += image.at<Vec3b>(ny, nx)[2] * filter[ky + 1][kx + 1];
                }
            }    
            result.at<Vec3b>(y, x)[0] = saturate_cast<uchar>(b);
            result.at<Vec3b>(y, x)[1] = saturate_cast<uchar>(g);
            result.at<Vec3b>(y, x)[2] = saturate_cast<uchar>(r);
        }
    }

    imshow("Original image", image);
    imshow("Result image", result);

    imwrite("salt_pepper_lpf.bmp", result);

    waitKey(0);
    return 0;
}