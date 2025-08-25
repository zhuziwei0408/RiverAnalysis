#include "InvadeThread.h"

#include "Analysis.h"
#include "DefineColor.h"

#include <glog/logging.h>

static int get_result(cv::Mat &srcImage, const cv::Scalar &color_people, const std::vector<cv::Rect> &detect_people, std::vector<cv::Rect> &result_rect_invade);
static std::vector<std::vector<cv::Point> > getObjectRect(cv::Mat &srcImg,const cv::Scalar &color);
static bool crossAlgorithm2(cv::Rect r1, cv::Rect r2);


InvadeThread::InvadeThread(Analysis* analysis_manager) 
    : manager(analysis_manager) {
        
}

int InvadeThread::LoadConfig(const AlgorithmConfig& input_config) {
    if (input_config.roi_rects_size() == 0) 
        return -1;
    config = input_config;
    return 0;
}

void InvadeThread::Run() {
    std::lock_guard<std::mutex> lk(_mutex);
    LOG(INFO) << "InvadeThread start";
    _is_run = true;
    uint32_t interval = config.detect_interval() * 1000;
    std::vector<cv::Rect> rects(config.roi_rects_size());
    for (size_t i = 0; i < rects.size(); ++i) {
        const AnalysisRect& rect = config.roi_rects(i);
        rects[i] = cv::Rect(rect.x(), rect.y(), rect.width(), rect.height());
    }
    while (_is_run) {
        cv::Mat origin_img = manager->GetOriginImg();
        cv::Mat segment_img = manager->GetSegmentImg();
        if (origin_img.empty() || segment_img.empty()) {
            usleep(interval);
            continue;
        }

#ifdef _DEBUG
        if (config.has_display() && config.display()) {
            std::string window_name = std::to_string(manager->Config().video_id());
            std::string origin_name = window_name + std::string("_Invade_Origin");
            std::string segment_name = window_name + std::string("_Invade_Segment");
            cv::imshow(origin_name, origin_img);
            cv::imshow(segment_name, segment_img);
            cv::waitKey(1);
        }
#endif
        std::vector<cv::Rect> result;
        int state = get_result(segment_img, PEOPLE_COLOR, rects, result);
        if (state == 0 && !result.empty()) 
            SetAlarm(true, result);
        else
            SetAlarm(false, result);
        usleep(interval);
    }
    LOG(INFO) << "InvadeThread end";
    _is_run = false;
    CallStop();
}

AnalysisAlarm InvadeThread::GetAlarm() {
    std::lock_guard<std::mutex> lk(alarm_mutex);
    AnalysisAlarm res = alarm;
    alarm.set_is_active(false);
    return res;
}

void InvadeThread::SetAlarm(bool is_active, const std::vector<cv::Rect>& result) {
    std::lock_guard<std::mutex> lk(alarm_mutex);
    alarm.set_is_active(is_active);
    alarm.clear_rects();
    if (!is_active)
        return;
    for (size_t i = 0; i < result.size(); ++i) {
        AnalysisRect* thisrect = alarm.add_rects();
        thisrect->set_x(result[i].x);
        thisrect->set_y(result[i].y);
        thisrect->set_width(result[i].width);
        thisrect->set_height(result[i].height);
    }
}

std::vector<std::vector<cv::Point>> getObjectRect(cv::Mat &srcImg,const cv::Scalar &color)
{
	cv::Mat segImg;
	std::vector<cv::Point> contourRect;
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;


	inRange(srcImg, color, color, segImg);//寻找颜色
	threshold(segImg, segImg, 200, 255, CV_THRESH_OTSU);//自适应二值化

	cv::Mat dstImage;
	findContours(segImg, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	if (contours.empty())
	{
		return std::vector<std::vector<cv::Point>>();
	}
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() > 50)
		{
			//绘制轮廓的最小外接矩形  
			cv::RotatedRect rect = minAreaRect(contours[i]);
			contourRect.push_back(rect.center);
			rectangle(srcImg, rect.boundingRect(), cv::Scalar(0, 0, 255));
		}
	}
	return contours;;
}
// 判断两矩形是否相交、原理狠简单、如果相交、肯定其中一个矩形的顶点在另一个顶点内、
bool crossAlgorithm2(cv::Rect r1, cv::Rect r2)
{
	int x1 = r1.x;
	int y1 = r1.y;
	int x2 = r1.x + r1.width;
	int y2 = r1.y + r1.height;

	int x3 = r2.x;
	int y3 = r2.y;
	int x4 = r2.x + r2.width;
	int y4 = r2.y + r2.height;

	return (((x1 >= x3 && x1 < x4) || (x3 >= x1 && x3 <= x2)) &&
		((y1 >= y3 && y1 < y4) || (y3 >= y1 && y3 <= y2))) ? true : false;
}

int get_result(cv::Mat &srcImage, const cv::Scalar &color_people, const std::vector<cv::Rect> &detect_people, std::vector<cv::Rect> &result_rect_invade)
{
	if (srcImage.empty())
	{
		return -1;
	}

	std::vector<std::vector<cv::Point>> RectPeople;
	cv::RotatedRect RectPeopleresult;
	bool judge;
	//入侵
	RectPeople = getObjectRect(srcImage, color_people);
	//还需要修改
	if (RectPeople.size() > 0)
	{
        for (size_t j = 0; j < detect_people.size(); ++j) 
        {
            for (size_t i = 0; i < RectPeople.size(); ++i)
		    {
			    RectPeopleresult = minAreaRect(RectPeople[i]);
			    judge = crossAlgorithm2(detect_people[j], RectPeopleresult.boundingRect());
			    if (judge)
			    {
				    result_rect_invade.push_back(RectPeopleresult.boundingRect());
			    }
		    }
        }        
	}
	return 0;
}


