#include "FloaterThread.h"
#include "Analysis.h"

#include "DefineColor.h"

#include <unistd.h>
#include <sys/time.h>
#include <glog/logging.h>

FloaterThread::FloaterThread(Analysis* analysis_manager) 
    : manager(analysis_manager) {

}

void FloaterThread::Run() {
    std::lock_guard<std::mutex> lk(_mutex);
    LOG(INFO) << "FloaterThread start";
    _is_run = true;
    uint32_t interval = config.detect_interval() * 1000;
    while(_is_run) {
        cv::Mat originMat = manager->GetOriginImg();
        cv::Mat segMat = manager->GetSegmentImg();
        cv::Mat foreground_img = manager->GetForegroundImg();

        if (originMat.empty() || segMat.empty() || foreground_img.empty()) {
            usleep(interval);
            continue;
        }

#ifdef _DEBUG
        if (config.has_display() && config.display()) {
            std::string window_name = std::to_string(manager->Config().video_id());
            std::string origin_name = window_name + std::string("_FloaterThread_origin");
            std::string segment_name = window_name + std::string("_FloaterThread_segment");
            cv::imshow(origin_name, originMat);
            cv::imshow(segment_name, segMat);
            cv::waitKey(1);
        }
#endif

        std::vector<cv::Rect> resultrect, resultrect_car;
        double speed = 0.0;
        int area = 0;
        int state = getfloater(segMat, foreground_img, FLOATER_COLOR, WATER_COLOR,CAR_COLOR, resultrect, speed, area, resultrect_car);
        if (state != 0 || resultrect.empty()) {
            usleep(interval);
            SetAlarm(false, 0, 0,resultrect);
            continue;
        } else {
		    double widthrate = (double)originMat.cols / segMat.cols;
            double heightrate = (double)originMat.rows / segMat.rows;
		    for (size_t i = 0; i < resultrect.size(); ++i) {
		        resultrect[i].x *= widthrate;
		        resultrect[i].width *= widthrate;
		        resultrect[i].y *= heightrate;
		        resultrect[i].height *= heightrate;
		    }
		    SetAlarm(true, area, speed,resultrect);
        }
        usleep(interval);
	}
    LOG(INFO) << "FloaterThread end";
    _is_run = false;
    CallStop();
}

AnalysisAlarm FloaterThread::GetAlarm() {
    std::lock_guard<std::mutex> lk(alarm_mutex);
    AnalysisAlarm res = alarm;
   // alarm.set_is_active(false);
    return res;
}

void FloaterThread::SetAlarm(bool is_active, float area, float speed,std::vector<cv::Rect>& floater_rect) {
    std::lock_guard<std::mutex> lk(alarm_mutex);
    alarm.set_is_active(is_active);
	alarm.clear_rects();
    if (!is_active) {
        alarm.set_floater_area(0);
        alarm.set_floater_speed(0);
    } else {
        alarm.set_floater_area(area);
        alarm.set_floater_speed(speed);
	    for(const cv::Rect& rect : floater_rect){
            AnalysisRect* newrect=alarm.add_rects();
            newrect->set_x(rect.x);
            newrect->set_y(rect.y);
            newrect->set_width(rect.width);
            newrect->set_height(rect.height);
        }
    }
}


std::vector<cv::Point>  FloaterThread::getObjectRect(cv::Mat &srcImg,const cv::Scalar &color,std::vector<cv::Rect> &result_floater, double& velocityvalue)
{
	cv::Mat segImg;
	std::vector<cv::Point> contourRectcenter, contourRect;
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	inRange(srcImg, color, color, segImg);

	threshold(segImg, segImg, 200, 255, CV_THRESH_OTSU);//自适应二值化
	cv::Mat dstImage;
	findContours(segImg, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() > 25)
		{
			//绘制轮廓的最小外接矩形  
			cv::RotatedRect rect = minAreaRect(contours[i]);
			cv::Point A, B, C, D;
			/*contourRectcenter.push_back(rect.center);*/
			rectangle(srcImg, rect.boundingRect(), cv::Scalar(0, 0, 255));
			A.x = rect.boundingRect().x;
			A.y = rect.boundingRect().y;
			B.x = rect.boundingRect().x + rect.boundingRect().width;
			B.y = rect.boundingRect().y ;
			C.x= rect.boundingRect().x;
			C.y = rect.boundingRect().y+rect.boundingRect().height;
			D.x = rect.boundingRect().x + rect.boundingRect().width;
			D.y = rect.boundingRect().y + rect.boundingRect().height;
			contourRect.push_back(A);
			contourRect.push_back(B);
			contourRect.push_back(C);
			contourRect.push_back(D);
			result_floater.push_back(rect.boundingRect());
			//旋转矩形
		}
	}
	return contourRect;
}
cv::Rect FloaterThread::RiverRect(cv::Mat srcImg,const cv::Scalar& color, std::vector<cv::Point>& contoursriverresult)
{
	cv::Mat segImg;
	std::vector<cv::Point> contourRect;
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	inRange(srcImg, color, color, segImg);
	cv::threshold(segImg, segImg, 200, 255, CV_THRESH_OTSU);
	cv::Mat riverimage = segImg;
	cv::Mat dstImage;
	int k = -1;
	cv::RotatedRect rectresult;

	findContours(segImg, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	if (contours.empty())
		return cv::Rect(0, 0, 0, 0);
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() > 50)
		{
			size_t maxarea = 100;
			if (contours[i].size() > maxarea)
			{
				maxarea = contours[i].size();
				k = i;
			}
		}
	}
	if (k == -1)
		return cv::Rect();
	contoursriverresult = contours[k];
	rectresult = minAreaRect(contoursriverresult);
	rectangle(srcImg, rectresult.boundingRect(), cv::Scalar(0, 0, 255));
	return rectresult.boundingRect();
}
int FloaterThread::getfloater(cv::Mat &segMat,cv::Mat &mod_input,const cv::Scalar &floater_color,const cv::Scalar &water_color, const cv::Scalar &car_color,std::vector<cv::Rect> &result, double& velocityvalue, int& totall,std::vector<cv::Rect> &result_car)
{
   std::vector<cv::Point> counter_max;
   cv::Rect river_Rect;
   river_Rect=RiverRect(segMat, water_color, counter_max);
    DetectFrameNumber_Floater--;
    if (DetectFrameNumber_Floater==0)
    {
	//原图背景建模
	std::vector<cv::Rect> result_mod,result_mod1;
	cv::Mat _rawMat,FStempMatforeground, PZtempMatforeground;
	result_mod.clear();
        result_mod1.clear();
        if ((river_Rect.width+ river_Rect.x<mod_input.cols)&&(river_Rect.height+river_Rect.y<mod_input.rows)&&river_Rect.x>=0 && river_Rect.y>=0&& river_Rect.width>mod_input.cols/2&& 	river_Rect.height>mod_input.rows/2)
	{
	_rawMat = mod_input(river_Rect);
	}
        else
	{
	_rawMat=mod_input.clone();
	}
	cv::erode(_rawMat, FStempMatforeground, cv::Mat(3, 3, CV_8U));
	cv::dilate(FStempMatforeground, PZtempMatforeground, cv::Mat(3, 3, CV_8U));
	cv::threshold(PZtempMatforeground, PZtempMatforeground, 125, 250, CV_THRESH_OTSU);
	std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Point> float_point_vec_mod;
	cv::findContours(PZtempMatforeground, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() >= 15)
		{
		//绘制轮廓的最小外接矩形  
			cv::RotatedRect rect_mod = minAreaRect(contours[i]);
			rectangle(mod_input, rect_mod.boundingRect(), cv::Scalar(0, 0, 255));
                        cv::Point A, B, C, D;
                        A.x = rect_mod.boundingRect().x;
			A.y = rect_mod.boundingRect().y;
			B.x = rect_mod.boundingRect().x + rect_mod.boundingRect().width;
			B.y = rect_mod.boundingRect().y ;
			C.x= rect_mod.boundingRect().x;
			C.y = rect_mod.boundingRect().y+rect_mod.boundingRect().height;
			D.x = rect_mod.boundingRect().x + rect_mod.boundingRect().width;
			D.y = rect_mod.boundingRect().y + rect_mod.boundingRect().height;
			float_point_vec_mod.push_back(A);
			float_point_vec_mod.push_back(B);
			float_point_vec_mod.push_back(C);
			float_point_vec_mod.push_back(D);
		result_mod1.push_back(rect_mod.boundingRect());
		}
	}
	if (float_point_vec_mod.size() != 0 && counter_max.size() != 0)
	{
        for (size_t j1 = 0; j1 < result_mod1.size(); ++j1) {
            
            cv::Rect& rect1 = result_mod1[j1];
            bool istrue1 = false;
            for (size_t k1 = 4*j1; k1 < 4*(j1+1); ++k1) {
		double ditance1 = pointPolygonTest(counter_max, float_point_vec_mod[k1], true);
                if (ditance1  > 0) 
		{
		   istrue1 = true;
                    break;
                }
            }
            if (istrue1) {
                result_mod.push_back(rect1);
            }
        }
	}
       //分割图像分析
	std::vector<cv::Rect> floaterrect;
	std::vector<cv::Point> float_point_vec = getObjectRect(segMat, floater_color, floaterrect, velocityvalue);
	getObjectRect(segMat, car_color, result_car, velocityvalue);
	result.clear();
       
	if (float_point_vec.size() != 0 && counter_max.size() != 0)
	{
        for (size_t j = 0; j < floaterrect.size(); ++j) {
            
            cv::Rect& rect = floaterrect[j];
            bool istrue = false;
            for (size_t k = 4*j; k < 4*(j+1); ++k) {
			    double ditance = pointPolygonTest(counter_max, float_point_vec[k], true);
                            //if(!counter_max_car.empty())
				//{
					 //double distance_car=pointPolygonTest(counter_max_car, float_point_vec[k], true);
				//}
                if (ditance > 0) 
		{
                    istrue = true;
                    break;
                }
            }
            if (istrue) {
                result.push_back(rect);
            }
        }
	}
        //综合分析
	if (result.size()== result_mod.size())
	{
		for (size_t m = 0; m < result.size(); ++m)
		{
			totall += result[m].width*result[m].height;;
		}
		if (!result.empty()&&!DetectFrame_Floater_1.empty())
		{
			DetectFrameNumber_Floater=5;
			return 0;
			
		}
		else
		{
                        DetectFrameNumber_Floater=5;
			return -1;

		}
	}
	else
	{
		if (result.size() > result_mod.size())
		{
			if (result_mod.size()!=0)
			{
				for (size_t m = 0; m < result_mod.size(); ++m)
				{
					totall += result_mod[m].width*result_mod[m].height;;
				}
				if (!result_mod.empty()&&!DetectFrame_Floater_1.empty())
				{
					DetectFrameNumber_Floater=5;
					return 0;
				}
				else
				{
					DetectFrameNumber_Floater=5;
					return -1;
				}
			}
			else
			{
				for (size_t m = 0; m < result.size(); ++m)
				{
					totall += result[m].width*result[m].height;
				}
				if (!result.empty()&&!DetectFrame_Floater_1.empty())
				{
					DetectFrameNumber_Floater=5;
					return 0;
				}
				else
				{
					DetectFrameNumber_Floater=5;
					return -1;
				}
			}
		}
		else
		{
			for (size_t m = 0; m < result.size(); ++m)
			{
				totall += result[m].width*result[m].height;
			}
			if (!result.empty()&&!DetectFrame_Floater_1.empty())
			{
				DetectFrameNumber_Floater=5;
				return 0;
			}
			else
			{
				DetectFrameNumber_Floater=5;
				return -1;
			}
		}
	}
    }
else if (DetectFrameNumber_Floater==1)
{
		if(!DetectFrame_Floater.empty())
{
		DetectFrame_Floater_1.clear();
		if (segMat.empty())
		{
			return -1;
		}
		//std::vector<cv::Point> RectPointPeople, RectPointFloat;
		std::vector<cv::Rect> RectPeople, RectFloat_1;
		//RectPointPeople = getObjectRect(segMat, color_people, RectPeople, 25);
		//RectPointFloat = getObjectRect(segMat, floater_color, RectFloat, 25);
	        std::vector<cv::Point> RectPointFloat_1 = getObjectRect(segMat, floater_color, RectFloat_1, velocityvalue);
             if (RectPointFloat_1.size() != 0 && counter_max.size() != 0)
	{
        for (size_t j = 0; j < RectFloat_1.size(); ++j) {
            
            cv::Rect& rect = RectFloat_1[j];
            bool istrue = false;
            for (size_t k = 4*j; k < 4*(j+1); ++k) {
			    double ditance = pointPolygonTest(counter_max, RectPointFloat_1[k], true);

                if (ditance > 0) 
		{
                    istrue = true;
                    break;
                }
            }
            if (istrue) {
                DetectFrame_Floater_1.push_back(rect);
            }
	}
}
}
                DetectFrameNumber_Floater=1;
                return 0;
}
else if(DetectFrameNumber_Floater==3)
{
  DetectFrameNumber_Floater=3;
  return 0;
}
else if(DetectFrameNumber_Floater==2)
{
  DetectFrameNumber_Floater=2;
  return 0;
}
         else 
	{
		DetectFrame_Floater.clear();
		//DetectFrame_People.clear();
		if (segMat.empty())
		{
			return -1;
		}
		//std::vector<cv::Point> RectPointPeople, RectPointFloat;
		std::vector<cv::Rect> RectPeople, RectFloat;
		//RectPointPeople = getObjectRect(segMat, color_people, RectPeople, 25);
		//RectPointFloat = getObjectRect(segMat, floater_color, RectFloat, 25);
	        std::vector<cv::Point> RectPointFloat = getObjectRect(segMat, floater_color, RectFloat, velocityvalue);
		
                if (RectPointFloat.size() != 0 && counter_max.size() != 0)
	{
        for (size_t j = 0; j < RectFloat.size(); ++j) {
            
            cv::Rect& rect = RectFloat[j];
            bool istrue = false;
            for (size_t k = 4*j; k < 4*(j+1); ++k) {
	     double ditance = pointPolygonTest(counter_max, RectPointFloat[k], true);

                if (ditance > 0) 
		{
                    istrue = true;
                    break;
                }
            }
            if (istrue) {
                DetectFrame_Floater.push_back(rect);
            }
		//DetectFrame_People = RectPeople;
	}
        }
        DetectFrameNumber_Floater = 4;
        return 0;
}
}

