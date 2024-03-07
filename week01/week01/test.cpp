#include <iostream>
#include "opencv2/opencv.hpp"

int main()
{
    std::cout << "Hello OpenCV " << CV_VERSION << std::endl;

    cv::Mat img;
    img = cv::imread("dog.jpg");

    if (img.empty()) {
        std::cerr << "image load failed!" << std::endl;
        return -1;
    }

    cv::namedWindow("image");
cv:imshow("image", img);

    cv::waitKey();
    return 0;
}