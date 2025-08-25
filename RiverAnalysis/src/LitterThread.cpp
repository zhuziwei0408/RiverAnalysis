#include "LitterThread.h"
#include "Analysis.h"
#include "DefineColor.h"

#include <unistd.h>
#include <glog/logging.h>

LitterThread::LitterThread(Analysis* analysis_manager)
    : manager(analysis_manager) {}

void LitterThread::Run() {
    std::lock_guard<std::mutex> lk(_mutex); 
    LOG(INFO) << "LitterThread start";
    _is_run = true;
    uint32_t interval = config.detect_interval() * 1000;
    while(_is_run) {
        cv::Mat originImg = manager->GetOriginImg();
        cv::Mat segmentImg = manager->GetSegmentImg();

        if (originImg.empty() || segmentImg.empty()) {
            usleep(interval);
            continue;
        }
#ifdef _DEBUG
        if (config.has_display() && config.display()) {
            std::string window_name = std::to_string(manager->Config().video_id());
            std::string origin_name = window_name + std::string("_Litter_origin");
            std::string segment_name = window_name + std::string("_Litter_segment");
            cv::imshow(origin_name, originImg);
            cv::imshow(segment_name, segmentImg);
            cv::waitKey(1);
        }
#endif

        int result = 0;
        std::vector<cv::Rect> people_rect, floater_rect,car_rect;
	bool is_car;
        int state = get_result(segmentImg, PEOPLE_COLOR, CAR_COLOR,FLOATER_COLOR, result, people_rect, floater_rect,car_rect,is_car);
        if (state == 0 && result == 1) {
            double widthrate = (double)originImg.cols / segmentImg.cols;
            double heightrate = (double)originImg.rows / segmentImg.rows;
            for (size_t i = 0; i < people_rect.size(); ++i) {
                people_rect[i].x *= widthrate;
                people_rect[i].width *= widthrate;
                people_rect[i].y *= heightrate;
                people_rect[i].height *= heightrate;
            }
            for (size_t i = 0; i < floater_rect.size(); ++i) {
                floater_rect[i].x *= widthrate;
                floater_rect[i].width *= widthrate;
                floater_rect[i].y *= heightrate;
                floater_rect[i].height *= heightrate;
            }
            SetAlarm(true, people_rect, floater_rect);
        } else {
            SetAlarm(false, people_rect, floater_rect);
        }
        usleep(interval);
    }
    LOG(INFO) << "LitterThread end";
    _is_run = false;
    CallStop();
}

AnalysisAlarm LitterThread::GetAlarm() {
    std::lock_guard<std::mutex> lk(alarm_mutex);
    AnalysisAlarm res = alarm;
    alarm.set_is_active(false);
    return res;
}

void LitterThread::SetAlarm(bool is_active, const std::vector<cv::Rect>& people_rect, const std::vector<cv::Rect>& floater_rect) {
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
    for (const cv::Rect& rect : floater_rect) {
        AnalysisRect* newrect = alarm.add_rects();
        newrect->set_x(rect.x);
        newrect->set_y(rect.y);
        newrect->set_width(rect.width);
        newrect->set_height(rect.height);
    }
}

std::vector<cv::Point> LitterThread::getObjectRect(cv::Mat &srcImg,const cv::Scalar &color, std::vector<cv::Rect>& obj_rect, uint32_t size)
{
	cv::Mat segImg;
	std::vector<cv::Point> contourRect;
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;


	inRange(srcImg, color, color, segImg);//寻找颜色
	threshold(segImg, segImg, 200, 255, CV_THRESH_OTSU);//自适应二值化
#ifdef _DEBUG
    if (config.has_display() && config.display()) {
        std::string window_name = std::to_string(manager->Config().video_id());
        std::string segImg_name = window_name + std::string("_Litter_SegImg2");
        cv::imshow(segImg_name, segImg);
        cv::waitKey(1);
    }
#endif
	cv::Mat dstImage;
	findContours(segImg, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
    obj_rect.clear();
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() > size)
		{
			//绘制轮廓的最小外接矩形  
			cv::RotatedRect rect = minAreaRect(contours[i]);
			contourRect.push_back(rect.center);
			rectangle(srcImg, rect.boundingRect(), cv::Scalar(0, 0, 255));
            obj_rect.push_back(rect.boundingRect());
		}
	}
	return contourRect;
}
int LitterThread::get_result(cv::Mat &srcImage, const cv::Scalar &color_people, const cv::Scalar &color_car,const cv::Scalar &color_floater,
                             int &result, std::vector<cv::Rect>& people_rect, std::vector<cv::Rect>& floater_rect,std::vector<cv::Rect>& car_rect, bool &is_car)
{
    DetectFrameNumber--;
    if (DetectFrameNumber==0)
    {
	if (srcImage.empty())
	{
		return -1;
	}
	std::vector<cv::Point> RectPointPeople, RectPointFloat,RectPointcar;
        std::vector<cv::Rect> car_rect;
	size_t distancePeopleFloat;
	//倾倒垃圾
	RectPointPeople = getObjectRect(srcImage, color_people, people_rect,30);
	RectPointFloat = getObjectRect(srcImage, color_floater, floater_rect,25);
        RectPointcar= getObjectRect(srcImage, color_car, car_rect,100);
	uint32_t minDistance = 400;
        is_car=false;
    for (size_t m=0; m < people_rect.size(); ++m)
	{
		for (size_t n=0; n < car_rect.size(); ++n)
		{
			bool isOverlapValue = isOverlap(people_rect[m], car_rect[n]);
			if (!isOverlapValue)
			{
				int dictancx = RectPointPeople[m].x - RectPointcar[n].x;
				int dictancy = RectPointPeople[m].y - RectPointcar[n].y;
				distancePeopleFloat = sqrt(dictancx*dictancx + dictancy*dictancy);
				if (distancePeopleFloat < minDistance)
				{
					minDistance = distancePeopleFloat;
				}
				if (50<minDistance&&minDistance< 100)
				{
  							is_car=true;
				}
				else
				{
                                        is_car=false;
				}
			}
			else
			{
                                is_car=false;
			}
		}
	}
    for (size_t m=0; m < people_rect.size(); ++m)
	{
		for (size_t n=0; n < floater_rect.size(); ++n)
		{
			bool isOverlapValue = isOverlap(people_rect[m], floater_rect[n]);
			if (!isOverlapValue)
			{
				int dictancx = RectPointPeople[m].x - RectPointFloat[n].x;
				int dictancy = RectPointPeople[m].y - RectPointFloat[n].y;
				distancePeopleFloat = sqrt(dictancx*dictancx + dictancy*dictancy);
				if (distancePeopleFloat < minDistance)
				{
					minDistance = distancePeopleFloat;
				}
				if (50<minDistance&&minDistance< 300)
				{
					if (DetectFrameFloater.size()!= floater_rect.size()&&!is_car)
						{
							result = 1;
							DetectFrameNumber = 3;
							return 0;
						}
						else
						{
							result = 0;
							DetectFrameNumber = 3;
							return 0;
						}
				}
				else
				{
					result = 0;
					DetectFrameNumber = 3;
					return 0;
				}
			}
			else
			{
				result = 0;
				DetectFrameNumber = 3;
				return 0;
			}
		}
	}
   }
      else
	{
		DetectFrameFloater.clear();
		DetectFramePeople.clear();
		if (srcImage.empty())
		{
			return -1;
		}
		std::vector<cv::Point> RectPointPeople, RectPointFloat;
		std::vector<cv::Rect> RectPeople, RectFloat;
		int distancePeopleFloat;
		bool isOverlapValue;
		//倾倒垃圾
		RectPointPeople = getObjectRect(srcImage, color_people, RectPeople, 25);
		RectPointFloat = getObjectRect(srcImage, color_floater, RectFloat, 25);
		int minDistance = 400;
		DetectFrameFloater = RectFloat;
		DetectFramePeople = RectPeople;
		DetectFrameNumber = 1;
	}
    return 0;
}

bool LitterThread::isOverlap(const cv::Rect &rc1, const cv::Rect &rc2) 
{                                                                            
	if (rc1.x + rc1.width  >= rc2.x &&                                         
		rc2.x + rc2.width  >=rc1.x &&                                          
		rc1.y + rc1.height >= rc2.y &&                                         
		rc2.y + rc2.height >= rc1.y                                            
		)                                                                      
		return true;                                                           
	else                                                                       
		return false;                                                          
}                                                                            
