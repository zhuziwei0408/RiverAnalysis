#include "WaterColorThread.h"
#include "Analysis.h"
#include "DefineColor.h"
#include <unistd.h>

#include <glog/logging.h>

WaterColorThread::WaterColorThread(Analysis* analysis_manager)
    :manager(analysis_manager) {
}

void WaterColorThread::Run() {
    std::lock_guard<std::mutex> lk(_mutex);
    LOG(INFO) << "WaterColorThread start";
    _is_run = true;
    uint32_t interval = config.detect_interval() * 1000;
    while(_is_run) {
        cv::Mat oriimg = manager->GetOriginImg();
        cv::Mat segmentimg = manager->GetSegmentImg();
        if(oriimg.empty() || segmentimg.empty()) {
            usleep(interval);
            continue;
        }
        cv::resize(oriimg, oriimg, cv::Size(segmentimg.cols, segmentimg.rows));    
#ifdef _DEBUG
        if (config.has_display() && config.display()) {
            std::string window_name = std::to_string(manager->Config().video_id());
            std::string origin_name = window_name + std::string("_Watercolor_origin");
            std::string segment_name = window_name + std::string("_Watercolor_segment");
            cv::imshow(origin_name, oriimg);
            cv::imshow(segment_name, segmentimg);
            cv::waitKey(1);
        }
#endif
        std::string result;
        int state = get_result_color(segmentimg, oriimg, WATER_COLOR, result);
        if (state == 1 || !result.empty()) {
            SetAlarm(result.c_str());
        }

        usleep(interval);
    }
    LOG(INFO) << "WaterColorThread stop";
    _is_run = false;
    CallStop();    
}

AnalysisAlarm WaterColorThread::GetAlarm() {
    std::lock_guard<std::mutex> lk(alarm_mutex);
    AnalysisAlarm res = alarm;
    alarm.set_is_active(false);
    return res;

}

void WaterColorThread::SetAlarm(const char* color) {
    std::lock_guard<std::mutex> lk(alarm_mutex);
    alarm.set_water_color(color);
    alarm.set_is_active(true);
}

void  WaterColorThread::rgbToHsv(double R, double G, double B, double& H, double& S, double& V)
{
	double MAX, MIN;
	MAX = B>G ? B : G;
	MAX = MAX>R ? MAX : R;
	MIN = B<G ? B : G;
	MIN = MIN<R ? MIN : R;

	if (MAX == MIN)
	{
		H = 0;
		S = 0;
		V = MAX / 255;
	}
	else
	{
		if (R == MAX)
		{
			H = (G - B) / (MAX - MIN);
		}
		if (G == MAX)
		{
			H = 2 + (B - R) / (MAX - MIN);
		}
		if (B == MAX)
		{
			H = 4 + (R - G) / (MAX - MIN);
		}
		H *= 60;
		while (H>360)
		{
			H -= 360;
		}
		while (H<0)
		{
			H += 360;
		}
		S = (MAX - MIN) / MAX;
		V = MAX / 255;
	}
}

std::string  WaterColorThread::getColor(double H, double S, double V)
{
	std::string ColorName = "未识别";
	//黑色
	if (V<30. / 255)
	{
		ColorName = "黑色";
	}
	else
	{
		//灰色
		if (S<0.1)
		{
			//白色
			if (V>180. / 255)
			{
				ColorName = "白色";
			}
			else
			{
				if (V>80. / 255)
				{
					//浅灰色
					ColorName = "浅灰色";
				}
				else
				{
					//深灰色
					ColorName = "深灰色";
				}
			}

		}
		else
		{
			//yellow(30-85)
			if (H <= 85 && H >= 25)
			{
				//深灰色
				if (S<0.4)
				{
					ColorName = "深灰色";
				}
				else
				{
					ColorName = "深灰色";
				}
			}
			else
			{
				//红色
				if (H<25 || H>330)
				{
					if (S<0.4)
					{
						ColorName = "浅红色";
					}
					else
					{
						ColorName = "深红色";
					}
				}
				else
				{
					//绿色
					if (H>85 && H <= 165)
					{
						if (S<0.4)
						{
							ColorName = "绿色";
						}
						else
						{
							ColorName = "绿色";
						}
					}
					else
					{
						//青色
						if (H>165 && H <= 205)
						{
							if (S<0.4)
							{
								ColorName = "青色";
							}
							else
							{
								ColorName = "青色";
							}
						}
						else
						{
							//蓝色
							if (H>205 && H <= 275)
							{
								if (S<0.4)
								{
									ColorName = "蓝色";
								}
								else
								{
									ColorName = "蓝色";
								}
							}
							//洋红色
							else
							{
								if (S<0.4)
								{
									ColorName = "洋红色";
								}
								else
								{
									ColorName = "洋红色";
								}
							}
						}
					}
				}
			}
		}
	}
	return ColorName;
}

std::string WaterColorThread::getWaterColor(const cv::Mat &segimg,const cv::Mat &img_origin,const cv::Scalar &color_river)
{
	if (img_origin.empty() || segimg.empty())
	{
		return 0;
	}
	cv::Mat seg_img, seg_th_img;
	inRange(segimg, color_river, color_river, seg_img);//寻找颜色
	threshold(seg_img, seg_th_img, 200, 255, CV_THRESH_OTSU);//自适应二值化
	unsigned int segRows = seg_th_img.rows;
	unsigned int segCols = seg_th_img.cols;
	unsigned int temSize = 0;


	unsigned int B = 0;
	unsigned int G = 0;
	unsigned int R = 0;

	for (uint segheight = 0; segheight < segRows; ++segheight)
	{
		for (uint segwidth = 0; segwidth < segCols; ++segwidth)
		{
			if (int gray = seg_th_img.at<uchar>(segheight, segwidth)==255)
			{

				B += img_origin.at<cv::Vec3b>(segheight, segwidth)[0];
				G += img_origin.at<cv::Vec3b>(segheight, segwidth)[1];
				R += img_origin.at<cv::Vec3b>(segheight, segwidth)[2];
				temSize++;
			}
		}
	}
	std::string _waterColor = "";
	if (temSize > 100)
	{
		B = B / temSize;
		G = G / temSize;
		R = R / temSize;
		double H = 0;
		double S = 0;
		double V = 0;
		rgbToHsv(R, G, B, H, S, V);
		_waterColor = getColor(H, S, V);
	}
	return _waterColor;
}
int WaterColorThread::get_result_color(cv::Mat &seg,cv::Mat &inimg , const cv::Scalar &color_river, std::string &result)
{
	if (seg.empty()|| inimg.empty())
	{
		return 0;
	}
	result = getWaterColor(seg ,inimg,color_river);
	return 1;
}

