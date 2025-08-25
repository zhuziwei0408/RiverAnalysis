#ifndef RIVER_ANALYSIS_LITTER_THREAD_H
#define RIVER_ANALYSIS_LITTER_THREAD_H

#include "RiverThread.h"
#include "opencv2/opencv.hpp"
class Analysis;

class LitterThread : public RiverThread {
public:
    LitterThread(Analysis* analysis_manager);
    ~LitterThread() {}

    void Run();

    AnalysisAlarm GetAlarm();
    void SetAlarm(bool is_active, const std::vector<cv::Rect>& people_rect, const std::vector<cv::Rect>& floater_rect);
    bool isOverlap(const cv::Rect& rc1, const cv::Rect& rc2);
private:
    LitterThread() { DetectFrameNumber = 3; }
    int get_result(cv::Mat &srcImage,const cv::Scalar &color_people,const cv::Scalar &color_car, const cv::Scalar &color_floater, int &result,
                   std::vector<cv::Rect>& people_rect, std::vector<cv::Rect>& floater_rect, std::vector<cv::Rect>& car_rect, bool &is_car);
    std::vector<cv::Point>  getObjectRect(cv::Mat &srcImg,const cv::Scalar &color, std::vector<cv::Rect>& obj_rect, uint32_t size);
    int DetectFrameNumber;
    std::vector<cv::Rect> DetectFrameFloater;
    std::vector<cv::Rect> DetectFramePeople;
    bool _isFirstDetect;//是否第一次检测
    cv::Mat preFreamBackground;//上一帧的背景

private:
    Analysis* manager;

};

#endif
