#include "FishingThread.h"

#include "Analysis.h"

#include "DefineColor.h"
#include <glog/logging.h>

FishingThread::FishingThread(Analysis* analysis_manager) 
    : manager(analysis_manager) {
}

void FishingThread::Run() {
    std::lock_guard<std::mutex> lk(_mutex);
    _is_run = true;
    LOG(INFO) << "FishingThread start";
    uint32_t interval = config.detect_interval() * 1000;
    while (_is_run) {
        cv::Mat origin_img = manager->GetOriginImg();
        cv::Mat segment_img = manager->GetSegmentImg();
        cv::Mat bkg_img = manager->GetForegroundImg();
        if (origin_img.empty() || segment_img.empty() || bkg_img.empty()) {
            usleep(interval);
            continue;
        }
#ifdef _DEBUG
        if (config.has_display() && config.display()) {
            std::string window_name = std::to_string(manager->Config().video_id());
            std::string origin_name = window_name + std::string("_Fishing_origin");
            std::string segment_name = window_name + std::string("_Fishing_segment");

            cv::imshow(origin_name, origin_img);
            cv::imshow(segment_name, segment_img);
            cv::waitKey(1);
        }
#endif
        std::vector<cv::Rect> people_rect;
        if (FishingEstimate(origin_img, segment_img, bkg_img))
            SetAlarm(true, people_rect);
        else
            SetAlarm(false, people_rect);
        usleep(interval);
    }
    LOG(INFO) << "FishingThread end";
    _is_run = false;
    CallStop();
}

AnalysisAlarm FishingThread::GetAlarm() {
    std::lock_guard<std::mutex> lk(alarm_mutex);
    AnalysisAlarm res = alarm;
    alarm.set_is_active(false);
    return res;
}

void FishingThread::SetAlarm(bool is_active, const std::vector<cv::Rect>& people_rect) {
    std::lock_guard<std::mutex> lk(alarm_mutex);
    alarm.set_is_active(is_active);
    alarm.clear_rects();
    if (!is_active)
        return;
    for (const cv::Rect& rect : people_rect) {
        AnalysisRect* newrect = alarm.add_rects();
        newrect->set_x(rect.x);
        newrect->set_y(rect.y);
        newrect->set_width(rect.width);
        newrect->set_height(rect.height);
    }
}

void FishingThread::ISFishing(cv::Mat fishingImg,double &lineAngle)
{
	if (fishingImg.empty())
		return ;
	IplImage houghImg = fishingImg;
	std::vector<cv::Point> binWhitePoint;
	LineFitting(&houghImg, binWhitePoint, lineAngle);

}

void FishingThread::TargetDectRect(cv::Mat segmentMat, cv::Scalar segmentColor, std::vector<cv::Point> &targetPoint, std::vector<cv::Rect>&targetRect)
{
	if (segmentMat.empty())
		return;
	cv::Mat segImg;
	inRange(segmentMat, segmentColor, segmentColor, segImg);//寻找颜色
	threshold(segImg, segImg, 200, 255, CV_THRESH_OTSU);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	std::vector<cv::Point> contourRect;
	cv::findContours(segImg, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() > 80) {
			cv::RotatedRect rect = minAreaRect(contours[i]);
			targetPoint.push_back(rect.center);
			targetRect.push_back(rect.boundingRect());
		}
		
	}
}

void FishingThread::PreProcessSrc(cv::Mat srcImg, cv::Mat &binMat)
{//getting edge 
	if (srcImg.empty())
		return;
	cv::Mat grayImage, enhanceImg, erzhihua;
	cv::cvtColor(srcImg, grayImage, CV_RGB2GRAY);
	cv::threshold(grayImage, grayImage, 20, 255, CV_THRESH_BINARY);
	bitwise_not(grayImage, binMat);
}

void FishingThread::DeleteHVLine(cv::Mat inputMat, cv::Mat& outMat)
{
	cv::Mat horizontalMat = inputMat.clone();
	cv::Mat verticalMat = inputMat.clone();
	int horizontalKernel = horizontalMat.cols / 16;
	int verticalKernel = verticalMat.rows / 16;
	if (verticalKernel == 0 || horizontalKernel == 0)
		return;

	cv::Mat horizontalLine = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(horizontalKernel, 1), cv::Point(-1, -1));
	cv::Mat verticalLine = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, verticalKernel), cv::Point(-1, -1));

	erode(horizontalMat, horizontalMat, horizontalLine, cv::Point(-1, -1));
	dilate(horizontalMat, horizontalMat, horizontalLine, cv::Point(-1, -1));

	erode(verticalMat, verticalMat, verticalLine, cv::Point(-1, -1));
	dilate(verticalMat, verticalMat, verticalLine, cv::Point(-1, -1));

	bitwise_not(horizontalMat, horizontalMat);//图像进行反转
	bitwise_not(verticalMat, verticalMat);
	cv::Mat tempMat;
	bitwise_and(inputMat, horizontalMat, tempMat);
	bitwise_and(tempMat, verticalMat, outMat);

}

void FishingThread::NextImgProcess(cv::Mat &binMat, double &lineAngle)
{
	//去除水平垂直线
	if (binMat.empty())
		return ;
	cv::Mat edgeOutMat;
	DeleteHVLine(binMat, edgeOutMat);
	if (edgeOutMat.empty())
		return ;

	//去除小点等
	cv::Mat romoveNoiseMat;
	IplImage inputImg = edgeOutMat;
	DeleteSmallArea(&inputImg, romoveNoiseMat);
	if (romoveNoiseMat.empty())
		return ;
	//直线拟合
	IplImage houghImg = romoveNoiseMat;
	std::vector<cv::Point> binWhitePoint;
	LineFitting(&houghImg, binWhitePoint, lineAngle);
}


bool FishingThread::FishingEstimate(cv::Mat &srcImg, cv::Mat segmentMat,cv::Mat foreGroundImg)
{
	if (srcImg.empty())
		return false;
	double widthRatio = (double)srcImg.cols / segmentMat.cols;
	double heightRatio = (double)srcImg.rows / segmentMat.rows;
	std::vector<cv::Point> peopleCenterPoint;
	std::vector<cv::Rect> peopleSegmentRect;
	std::vector<cv::Point> riverCenterPoint;
	std::vector<cv::Rect> riverSegmentRect;
	TargetDectRect(segmentMat, PEOPLE_COLOR, peopleCenterPoint, peopleSegmentRect);
	TargetDectRect(segmentMat, WATER_COLOR, riverCenterPoint, riverSegmentRect);
	if (peopleCenterPoint.size() == 0 || riverCenterPoint.size() == 0)
		return false;

	cv::Point riverLocation(0, 0);
	int maxArea = 0;
	for (size_t j = 0; j < riverCenterPoint.size(); ++j)
	{
		if (riverSegmentRect[j].height * riverSegmentRect[j].width > maxArea)
		{
			maxArea = riverSegmentRect[j].height * riverSegmentRect[j].width;
			riverLocation.x = riverCenterPoint[j].x;
			riverLocation.y = riverCenterPoint[j].y;
		}
	}
	if (riverLocation.x <= 0 || riverLocation.y <= 0|| riverLocation.x >= segmentMat.cols||riverLocation.y>=segmentMat.rows)
		return false;
	for (size_t i = 0; i < peopleCenterPoint.size(); ++i)
	{
		cv::Rect fishingRodRect;
		cv::Rect peopleRealRect;
		cv::Rect riverRealRect;

		peopleRealRect.x = peopleSegmentRect[i].x * widthRatio;
		peopleRealRect.y = peopleSegmentRect[i].y * heightRatio;
		peopleRealRect.width = peopleSegmentRect[i].width * widthRatio;
		peopleRealRect.height = peopleSegmentRect[i].height * heightRatio;
		if (peopleRealRect.x <= 0||peopleRealRect.y<=0|| peopleRealRect.y>srcImg.rows||peopleRealRect.x>srcImg.cols)
			continue;
		if (peopleCenterPoint[i].x > riverLocation.x)//河左人右
		{
			fishingRodRect.y = peopleRealRect.y *0.9;
			fishingRodRect.x = peopleRealRect.x - peopleRealRect.width * 2;
			if (fishingRodRect.x <= 0)
			{
				fishingRodRect.x = riverLocation.x*widthRatio;
				fishingRodRect.width = abs(peopleSegmentRect[i].x - riverLocation.x)*widthRatio;
			}
			else
				fishingRodRect.width = peopleRealRect.width * 2 - 2;

			fishingRodRect.height = peopleRealRect.height * 1.5;
			if (fishingRodRect.height + fishingRodRect.y>srcImg.rows)
				continue;
			cv::Mat fishingRodDetectRegion = srcImg(fishingRodRect);
			rectangle(srcImg, peopleRealRect, cv::Scalar(0, 0, 255));

			//高斯背景建模进行前景提取
			if (foreGroundImg.empty())
				return false;
			cv::Mat foreImg = foreGroundImg;
			cv::Rect rectForeGround = cv::Rect(fishingRodRect.x /2, fishingRodRect.y /2, peopleRealRect.width/2, peopleRealRect.height/2);
			if(abs(fishingRodRect.x / 2 - peopleRealRect.width / 2)<0|| abs(fishingRodRect.y / 2+ peopleRealRect.height / 2)>foreImg.rows)
				continue;
			foreImg = foreImg(rectForeGround);
			double lineAngleBgModel = 0.0;
			ISFishing(foreImg, lineAngleBgModel);
			
			//二值化进行直线的提取
			double lineAngleBin = 0.0;
			cv::Mat binMat;
			PreProcessSrc(fishingRodDetectRegion, binMat);
			NextImgProcess(binMat, lineAngleBin);

			if (abs(lineAngleBgModel - lineAngleBin) < 4 && 
				abs(lineAngleBgModel - lineAngleBin) > 0&&
				lineAngleBgModel!=0&& lineAngleBin!=0)
			{
				return true;
			}
			
		}
		if (peopleCenterPoint[i].x < riverLocation.x)//人左河右
		{
			fishingRodRect.x = peopleRealRect.x + peopleRealRect.width ;
			if(fishingRodRect.x>srcImg.cols)
				fishingRodRect.x = peopleRealRect.x;
			fishingRodRect.y = peopleRealRect.y*1.2;
			if(fishingRodRect.y>srcImg.rows)
				fishingRodRect.y=peopleRealRect.y;
			fishingRodRect.width = peopleRealRect.width * 2 - 2;
			if (fishingRodRect.x + fishingRodRect.width > srcImg.cols)
				continue;
				
			fishingRodRect.height = peopleRealRect.height * 1.5;
			if (fishingRodRect.height + fishingRodRect.y > srcImg.rows)
				continue;
			cv::Mat fishingRodDetectRegion = srcImg(fishingRodRect);
			if (fishingRodDetectRegion.empty())
				continue ;
			rectangle(srcImg, peopleRealRect, cv::Scalar(0, 0, 255));

			//高斯背景建模进行前景提取
			if (foreGroundImg.empty())
				continue;
			cv::Mat foreImg = foreGroundImg;
			cv::Rect rectForeGround = cv::Rect(fishingRodRect.x / 2, fishingRodRect.y / 2, peopleRealRect.width / 2, peopleRealRect.height / 2);
			if(fishingRodRect.x / 2 + peopleRealRect.width / 2>foreImg.cols|| fishingRodRect.y / 2+ peopleRealRect.height / 2>foreImg.rows)
				continue;
			foreImg = foreImg(rectForeGround);
			if (foreImg.empty())
				continue ;
			double lineAngleBgModel = 0.0;
			ISFishing(foreImg, lineAngleBgModel);

			//二值化进行直线的提取
			double lineAngleBin = 0.0;
			cv::Mat binMat;
			PreProcessSrc(fishingRodDetectRegion, binMat);
			NextImgProcess(binMat, lineAngleBin);
			if (abs(lineAngleBgModel - lineAngleBin) < 4 &&
				abs(lineAngleBgModel - lineAngleBin) > 0 &&
				lineAngleBgModel != 0 && lineAngleBin != 0)
			{
				return true;
			}
			
		}
	}
	return false;
}

void FishingThread::DeleteSmallArea(IplImage* inputImage, cv::Mat &outputImage)
{//判断图是否具有白色点，没有则返回原图
	CvSeq* contour = NULL;
	double tmparea = 0.0;
	uchar *pp;
	CvMemStorage* storage = cvCreateMemStorage(0);
	IplImage* img_Clone = inputImage;
	cvFindContours(img_Clone, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, cvPoint(0, 0));
	//开始遍历轮廓树         
	CvRect rect;
	cvDrawContours(img_Clone, contour, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 1, CV_FILLED, 8, cvPoint(0, 0));
	while (contour)
	{
		tmparea = fabs(cvContourArea(contour));
		rect = cvBoundingRect(contour, 0);
		if (tmparea >= 10)
		{
			contour = contour->h_next;
			continue;
		}
		//当连通域的中心点为黑色时，而且面积较小则用白色进行填充
		pp = (uchar*)(img_Clone->imageData + img_Clone->widthStep*(rect.y + rect.height / 2) + rect.x + rect.width / 2);
		if (pp[0] != 0)
		{
			contour = contour->h_next;
			continue;
		}
		for (int y = rect.y; y <= rect.y + rect.height; y++)
		{
			for (int x = rect.x; x <= rect.x + rect.width; x++)
			{
				pp = (uchar*)(img_Clone->imageData + img_Clone->widthStep*y + x);
				if (pp[0] == 255)
				{
					pp[0] = 0;
				}
			}
		}
		contour = contour->h_next;
	}

	int countPoint = 0;
	std::vector<cv::Point> binWhitePoint;
	GetBinWhitePoint(img_Clone, binWhitePoint, countPoint);
	if (countPoint < 10)
		outputImage = cv::cvarrToMat(inputImage, false);
	else
		outputImage = cv::cvarrToMat(img_Clone, false);
	cvReleaseMemStorage(&storage);
}

void FishingThread::GetBinWhitePoint(IplImage* inputImage, std::vector<cv::Point>& binWhitePoint, int &countPint)
{//返回一个vector容器
	cv::Mat src = cv::cvarrToMat(inputImage, false);
	if (src.empty())
		return;
	for (size_t i = 10; i < src.rows - 1; i++)
	{
		const uchar* pixelPtr = src.ptr<uchar>(i);
		for (int j = 10; j < src.cols -1 ; j++)
		{
			if (pixelPtr[j] >100)
			{
				binWhitePoint.push_back(cv::Point(j, i));
				++countPint;
			}
		}
	}
}

void FishingThread::LineFitting(IplImage* inputImage, std::vector<cv::Point>& binWhitePoint, double &lineAngle)
{
	int countpoint = 0;
	GetBinWhitePoint(inputImage, binWhitePoint, countpoint);
	if (countpoint <= 2 || countpoint>70|| binWhitePoint.size()==0)
		return ;
	cv::Mat dst = cv::cvarrToMat(inputImage);
	cv::Mat blankImage = cv::Mat::zeros(dst.rows, dst.cols, CV_8UC1);
	cv::Vec4f line_para;
	cv::fitLine(binWhitePoint, line_para, CV_DIST_HUBER, 0, 1e-2, 1e-2);

	//最小二乘拟合计算直线的倾角
	size_t pointCount = binWhitePoint.size();
	if (pointCount > 0)
	{
		size_t xCount = 0;
		size_t yCount = 0;
		size_t xyCount = 0;
		size_t xxCount = 0;
		for (size_t i = 0; i < pointCount; i++)
		{
			xCount += binWhitePoint[i].x;
			yCount += binWhitePoint[i].y;
			xyCount += binWhitePoint[i].x * binWhitePoint[i].y;
			xxCount += binWhitePoint[i].x * binWhitePoint[i].x;
		}
		double k = (double)(pointCount * xyCount - xCount * yCount) / (double)(pointCount * xxCount - xCount * xCount);
		double sinValue = -k / (sqrt(1 + k * k));
		double radian = asin(sinValue);
		double pi = 3.1415926535;

		lineAngle = radian * 180.0 / pi;
#ifdef _DEBUG
        if (config.has_display() && config.display()) {
            //std::string window_name = std::to_string(manager->Config().video_id());
            //std::string line_name = window_name + std::string("_Fishing_line");
		    //imshow(line_name, blankImage);
        }
#endif
	}
}

