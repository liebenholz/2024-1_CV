#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main() {
    // cout << "Hello OpenCV " << CV_VERSION << endl;

    Mat result_image, image;
    image = imread("cat.bmp", IMREAD_COLOR);
    image.copyTo(result_image);

    if (image.empty()) {
        cout << "image load failed!" << endl;
        return -1;
    }

    /*
    int i = 0, j = 0;

    cout << "Pixel Value:" << image.at<Vec3b>(j, i) << " ";
    cout << "B:" << (unsigned int) image.at<Vec3b>(j, i)[0] << " ";
    cout << "G:" << (unsigned int) image.at<Vec3b>(j, i)[1] << " ";
    cout << "R:" << (unsigned int) image.at<Vec3b>(j, i)[2] << "\n";

    cout << "Grayscale:" << (unsigned int) image.at<uchar>(j, i) << "\n";
    */

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int b = 255 - image.at<Vec3b>(y, x)[0];
            int g = 255 - image.at<Vec3b>(y, x)[1];
            int r = 255 - image.at<Vec3b>(y, x)[2];

            result_image.at<Vec3b>(y, x)[0] = saturate_cast<uchar>(b);
            result_image.at<Vec3b>(y, x)[1] = saturate_cast<uchar>(g);
            result_image.at<Vec3b>(y, x)[2] = saturate_cast<uchar>(r);
        }
    }

    imshow("Original image", image);
    imshow("Inverted image", result_image);
    imwrite("cat_inverted.jpg", result_image);
    
    waitKey(0);

    return 0;
}