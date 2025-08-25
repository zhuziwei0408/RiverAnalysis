#ifndef RIVER_ANALYSIS_FLOATER_THREAD_H
#define RIVER_ANALYSIS_FLOATER_THREAD_H

#include "RiverThread.h"

#include "opencv2/opencv.hpp"
#include <vector>



class Analysis;

class FloaterThread : public RiverThread {
public:
    FloaterThread(Analysis* analysis_manager);
    ~FloaterThread() {}
    void Run();

    AnalysisAlarm GetAlarm();
    void SetAlarm(bool is_active, float area, float speed, std::vector<cv::Rect>& floater_rect);

private:
    FloaterThread() { DetectFrameNumber_Floater = 5; }
    std::vector<cv::Point>  getObjectRect(cv::Mat &srcImg,const cv::Scalar &color,std::vector<cv::Rect> &result_floater, double& velocityvalue);
    int getfloater(cv::Mat &segMat,cv::Mat &mod_input,const cv::Scalar &floater_color,const cv::Scalar &water_color, const cv::Scalar &car_color,std::vector<cv::Rect> &result, double& velocityvalue,   
int& totall,std::vector<cv::Rect> &result_car);
    cv::Rect RiverRect(cv::Mat srcImg,const cv::Scalar& color, std::vector<cv::Point>& contoursriverresult);
    int DetectFrameNumber_Floater;
    std::vector<cv::Rect> DetectFrame_Floater,DetectFrame_Floater_1;
    std::vector<cv::Rect> DetectFrame_People;
    bool _isFirstDetect;//是否第一次检测
    cv::Mat preFreamBackground;//上一帧的背景

private:
    Analysis* manager;
};

#endif
