#include "DefineColor.h"

static const std::vector<cv::Scalar> colors {
    cv::Scalar(0, 0, 0), // 背景
    cv::Scalar(127,0,0), // 鱼竿 
    cv::Scalar(255,0,0), // 人
    cv::Scalar(0,127,0), // 水面
    cv::Scalar(0,255,0), // 垃圾
    cv::Scalar(0,0,127), // 水尺
    cv::Scalar(0,0,255)  // 车
};

const cv::Scalar& get_color_of_label(uint8_t index) {
    if (index >= colors.size())
        return colors[0];
    return colors[index];
}

