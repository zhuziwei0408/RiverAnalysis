
#ifndef RIVER_ANALYSIS_DEFINE_COLOR_H
#define RIVER_ANALYSIS_DEFINE_COLOR_H

#include <opencv2/opencv.hpp>

#define BACKGROUND_COLOR      get_color_of_label(0)
#define FISHING_COLOR         get_color_of_label(1)
#define PEOPLE_COLOR          get_color_of_label(2)
#define WATER_COLOR           get_color_of_label(3)
#define FLOATER_COLOR         get_color_of_label(4)
#define WATERGAUGE_COLOR      get_color_of_label(5)
#define CAR_COLOR             get_color_of_label(6)

const cv::Scalar& get_color_of_label(uint8_t index);

#endif
