#ifndef RIVER_ANALYSIS_WATERCOLOR_THREAD_H
#define RIVER_ANALYSIS_WATERCOLOR_THREAD_H

#include "RiverThread.h"
#include "opencv2/opencv.hpp"

class Analysis;

class WaterColorThread : public RiverThread {
public:
    WaterColorThread(Analysis* analysis_manager);
    ~WaterColorThread() {}

    void Run();

    AnalysisAlarm GetAlarm();
    void SetAlarm(const char* color);

private:
	int get_color(cv::Mat &srcImage,const cv::Scalar &color_river,  int &resultt);
	void rgbToHsv(double R, double G, double B, double& H, double& S, double& V);
	std::string getWaterColor(const cv::Mat &segimg, const cv::Mat &img_origin, const cv::Scalar &color_river);
	std::string  getColor(double H, double S, double V);
	int get_result_color(cv::Mat &seg, cv::Mat &inimg, const cv::Scalar &color_river, std::string &result);
	cv::Rect RiverRect(cv::Mat srcImg,const cv::Scalar& color, std::vector<cv::Point>& contoursriverresult);
private:
    Analysis* manager;
};
#endif
