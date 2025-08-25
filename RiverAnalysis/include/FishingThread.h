#ifndef RIVER_ANALYSIS_FISHING_THREAD_H
#define RIVER_ANALYSIS_FISHING_THREAD_H

#include "RiverThread.h"
#include <opencv2/opencv.hpp>

class Analysis;
class FishingThread : public RiverThread {
public:
    FishingThread(Analysis* analysis_manager);
    ~FishingThread() {}

    void Run();

    AnalysisAlarm GetAlarm();
    void SetAlarm(bool is_active, const std::vector<cv::Rect>& people_rect);
private:
	void NextImgProcess(cv::Mat &binMat, double &lineAngle);
	void PreProcessSrc(cv::Mat srcImg, cv::Mat &binMat);
	void DeleteHVLine(cv::Mat inputMat, cv::Mat& outMat);

	void ISFishing(cv::Mat fishingImg, double &lineAngle);
	void GetBinWhitePoint(IplImage* inputImage, std::vector<cv::Point>& binWhitePoint, int &countPint);
	void DeleteSmallArea(IplImage* inputImage, cv::Mat &outputImage);
	void TargetDectRect(cv::Mat segmentMat, cv::Scalar segmentColor, std::vector<cv::Point> &targetPoint, std::vector<cv::Rect> &targetRect);

	void LineFitting(IplImage* inputImage, std::vector<cv::Point>& binWhitePoint, double &lineAngle);
	bool FishingEstimate(cv::Mat &srcImg, cv::Mat segmentMat, cv::Mat bgMat);

private:
    Analysis* manager;
};

#endif
