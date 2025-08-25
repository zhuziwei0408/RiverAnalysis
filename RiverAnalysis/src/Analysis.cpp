#include "Analysis.h"

#include "TensorflowThread.h"
#include "WaterColorThread.h"
#include "FloaterThread.h"
#include "LitterThread.h"
#include "AlarmMsgQueue.h"
#include "InvadeThread.h"
#include "FishingThread.h"
#include "WaterGaugeThread.h"
#include "DefineColor.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <glog/logging.h>
 
using namespace cv;  
using namespace std;

static int GetAlgorithmSceneType(AlgorithmType type) {
    switch(type) {
    case SEGMANTIC:
        return -1;
    case WATERGAUGE:
        return 2;
    case WATERCOLOR:
        return 8;
    case INVADE:
        return 9;
    case FLOATER:
        return 1;
    case FISHING:
        return 5;
    case LITTER:
        return 3;
    case SWIMING:
        return 6;
    }
    return -1;
}

Analysis::Analysis()
    : send_client(NULL), send_queue(NULL){
}

Analysis::~Analysis() {
    for (RiverThread*& thread : algorithms) {
        thread->Stop();
        delete thread;
        thread = NULL;
    }
    algorithms.clear();
    puttext.release();
    
    if (send_client) {
        send_client->Stop();
        delete send_client;
        send_client = NULL;
    }
    if (send_queue) {
        delete send_queue;
        send_queue = NULL;
    }
}

RiverThread* Analysis::GetAlgorithm(const AlgorithmConfig& config) {
    RiverThread* inst = NULL;
    switch(config.algorithm_type()) {
        case SEGMANTIC:
            inst = new TensorflowThread(this);
            break;
        case WATERGAUGE:
            inst = new WaterGaugeThread(this);
            break;
        case WATERCOLOR:
            inst = new WaterColorThread(this);
            break;
        case INVADE:
            inst = new InvadeThread(this);
            break;
        case FLOATER:
            inst = new FloaterThread(this);
            break;
        case FISHING:
            inst = new FishingThread(this);
            break;
        case LITTER:
            inst = new LitterThread(this);
            break;
        case SWIMING:
            break;
    }

    if (inst != NULL && inst->LoadConfig(config) != 0) {
        delete inst;
        inst = NULL;
    }
    return inst;
}

int Analysis::LoadConfig(const char* config_path/*./config/1.config*/) {
    if (config_path == NULL)
        return -1;
    int fd = open(config_path, O_RDONLY);
    if (fd < 0) {
        LOG(ERROR) << "Config not exist: " << config_path;
        return -1;
    }
    google::protobuf::io::FileInputStream fileinput(fd);
    fileinput.SetCloseOnDelete(true);
    if (!google::protobuf::TextFormat::Parse(&fileinput, &config)) {
        LOG(ERROR) << "protobuf parse failed: " << config_path;
        return -1;
    }

    if (puttext.Loadttc(config.ttc_path().c_str()) != 0) {
        LOG(ERROR) << "Load font ttc failed: " << config.ttc_path();
        return -1;
    }
    gauss_bg_modeling =new GaussBgModeling(_mod);
    //每一个分析实例都需有自己的发送消息实例
    send_queue = new AlarmMsgQueue(50, 1000);
    send_client = new HttpClient(send_queue, config.send_url());
    //为每一类检测算法new出一个实例并推进容器
    for (int i = 0; i < config.algorithms_size(); ++i) {
        const AlgorithmConfig& al_config = config.algorithms(i);
        RiverThread* inst = GetAlgorithm(al_config);
        if (inst != NULL)
            algorithms.push_back(inst);
    }
    return 0;
}

void Analysis::Run() {
   
    LOG(INFO) << "Analysis Start";
    uint32_t interval = config.detect_interval() * 1000;
    _is_run = true;
    uint32_t retrytime = 0;
    cv::Mat img;
//luzhi
    cv::VideoCapture cap;
    // cv::VideoWriter writer;
    // static int video_index = 1;

    if (algorithms.empty()) {
        LOG(ERROR) << "No algorithm";
        goto _thread_end;
    }
    
    cap.open(config.input_url()); 
    if (!cap.isOpened()) {
        if (retrytime > 3) {
            LOG(ERROR) << "cap can not open: " << config.input_url();   
            goto _thread_end;
        }
        cap.open(config.input_url());
        retrytime++;
    }
 //luzhi
    // writer.open(std::to_string(video_index) + std::string("record.avi"), CV_FOURCC('M', 'J', 'P', 'G'), 25, Size(1280, 720), true);
    // video_index++;

    //开启每一类算法
    StartAnalysis();
    retrytime = 0;

    gettimeofday(&last_time, NULL);
    while (_is_run) {
        cap >> img;
        if (img.empty() || !cap.isOpened()) {
            LOG(WARNING) << "Empty img";
            if (retrytime > 10) {
                LOG(WARNING) << "retrytime > 10";
                break;
            }
            if (!cap.isOpened()) {
                LOG(WARNING) << "cap is close";
                cap.release();
                cap.open(config.input_url());
            }
            retrytime++;
            continue;
        }
        retrytime = 0;
        SetOriginImg(img);
        if (config.has_open_modeling() && config.open_modeling()) {
            SetForegroundImg(gauss_bg_modeling->GetForegroundImg(img,_mod));
        }
        DrawInfo(img, segment_img);
//
	    // writer.write(img);
        usleep(interval);
    }
    LOG(ERROR) << "Analysis break";
    StopAnalysis();
    cv::destroyAllWindows();
_thread_end:
    _is_run = false;
    CallStop();
    return;
}

void Analysis::DrawInfo(cv::Mat& img,cv::Mat segmentImg) {
    std::vector<AnalysisAlarm> alarms;
    uint16_t xpos = 20, ypos = 20;
    // 循环获取各算法结果并叠加发送
    for (RiverThread* ptr : algorithms) {
        if (ptr == NULL)
            continue;
        auto type = ptr->config.algorithm_type();
     //   if (type == SEGMANTIC){
        //   continue;
       // }
          
        AnalysisAlarm alarm = ptr->GetAlarm();
        char appendtext[100] = {0};
        float area = 0.0, speed = 0.0;
        cv::Mat segImg;
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        switch (type) {
        case SEGMANTIC:
                if(segmentImg.empty())
                {
                    break;
                }
                cv::inRange(segmentImg, PEOPLE_COLOR, PEOPLE_COLOR, segImg);
               // cv::imshow("inRange",segImg);
                cv::threshold(segImg, segImg, 200, 255, CV_THRESH_OTSU);
                findContours(segImg, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
                for (size_t i = 0; i < contours.size(); i++){
                     if (contours[i].size() > 25){
	                 cv::RotatedRect rect = minAreaRect(contours[i]);
                         double ratioX=(double)img.cols/(double)segImg.cols;
                         double ratioY=(double)img.rows/(double)segImg.rows;
                         cv::Rect bounding_rect;
                         bounding_rect.x=rect.boundingRect().x*ratioX;
                         bounding_rect.y=rect.boundingRect().y*ratioY;
                         bounding_rect.width=rect.boundingRect().width*ratioX;
                         bounding_rect.height=rect.boundingRect().height*ratioY;
                         rectangle(img, bounding_rect, cv::Scalar(0, 0, 255));
                         int tx_pos=rect.boundingRect().x*ratioX;
                         int ty_pos=rect.boundingRect().y*ratioY-18;
                         puttext.putText(img, "人", cv::Point(tx_pos, ty_pos), cv::Scalar(255,0,0), 15);
                      }
                }
              //  cv::waitKey(10);
                break;
        case WATERGAUGE:
                snprintf(appendtext, 100, "水尺检测：%.2f", alarm.water_gauge_num());
                alarm.set_scene_type(WATERGAUGE);
                if (alarm.is_active()) {
                    alarms.push_back(alarm);
                    LOG(INFO) << appendtext;
                }
if(img.cols>2000)
{
                puttext.putText(img, appendtext, cv::Point(xpos, ypos), cv::Scalar(0,0,255), 60);
}
else
{
puttext.putText(img, appendtext, cv::Point(xpos, ypos), cv::Scalar(0,0,255), 20);
}
                ypos += 40*1.5;
                break;
        case WATERCOLOR:
                snprintf(appendtext, 100, "水色检测：%s", alarm.water_color().c_str());
                alarm.set_scene_type(WATERCOLOR);
                if (alarm.is_active()) {
                    alarms.push_back(alarm);
                    LOG(INFO) << appendtext;
                }
if(img.cols>2000)
{
                puttext.putText(img, appendtext, cv::Point(xpos, ypos), cv::Scalar(0,0,255), 60);
}
else
{
     puttext.putText(img, appendtext, cv::Point(xpos, ypos), cv::Scalar(0,0,255), 20);
}
                ypos += 20*1.5;
                break;
        case SWIMING:
                 alarm.set_scene_type(SWIMING);
                break;
        case LITTER:
                snprintf(appendtext, 100, "倾倒垃圾： %s", alarm.is_active()?"是":"否");
                if (alarm.is_active()) {
                    alarm.set_scene_type(LITTER);
                    alarms.push_back(alarm);
                    LOG(INFO) << appendtext;
                }
                for (int i = 0; i < alarm.rects_size(); ++i) {
                    AnalysisRect rect = alarm.rects(i);
                    cv::rectangle(img, cv::Rect(rect.x(), rect.y(), rect.width(), rect.height()), cv::Scalar(0,0,255));
                    puttext.putText(img, "垃圾", cv::Point(rect.x(), rect.y()-18), cv::Scalar(255,0,0),15);
                }
                puttext.putText(img, appendtext, cv::Point(xpos, ypos), cv::Scalar(255,0,0),20);
                ypos += 20*1.5;
                break;
        case INVADE:
                snprintf(appendtext, 100, "入侵检测： %s", alarm.is_active()?"是":"否");
                if (alarm.is_active()) {
                    alarm.set_scene_type(INVADE);
                    alarms.push_back(alarm);
                    LOG(INFO) << appendtext;
                }
               
                for (int i = 0; i < alarm.rects_size(); ++i) {
                    AnalysisRect rect = alarm.rects(i);
                    cv::rectangle(img, cv::Rect(rect.x(), rect.y(), rect.width(), rect.height()), cv::Scalar(0,0,255));
                }

                for (int roi_index = 0; roi_index < ptr->config.roi_rects_size(); ++roi_index) {
                    AnalysisRect rect = ptr->config.roi_rects(roi_index);
                    //cv::rectangle(img, cv::Rect(rect.x(), rect.y(), rect.width(), rect.height()), cv::Scalar(0,255,0));
                }
                puttext.putText(img, appendtext, cv::Point(xpos, ypos), cv::Scalar(255,0,0), 20);
                ypos += 20*1.5;
                break;
        case FLOATER:
                area = alarm.floater_area();
                speed = alarm.floater_speed();
                alarm.set_scene_type(FLOATER);
                if (alarm.is_active()) {
               alarms.push_back(alarm);
                 LOG(INFO) << "漂浮物面积: " << area << "流速: " << speed;
                }
                snprintf(appendtext, 100, "漂浮物面积： %.2f 流速： %.2f", area, speed);
                for (int i = 0; i < alarm.rects_size(); ++i) {
                    AnalysisRect rect = alarm.rects(i);
                    cv::rectangle(img, cv::Rect(rect.x(), rect.y(), rect.width(), rect.height()), cv::Scalar(0,0,255));
                    puttext.putText(img, "漂浮物", cv::Point(rect.x(), rect.y()-18), cv::Scalar(255,0,0),15);
                }
                puttext.putText(img, appendtext, cv::Point(xpos, ypos), cv::Scalar(255,0,0), 20);
                ypos += 20*1.5;
                break;
        case FISHING:
                snprintf(appendtext, 100, "钓鱼检测： %s", alarm.is_active()?"是":"否");
                 alarm.set_scene_type(FISHING);
                if (alarm.is_active()) {
                    alarms.push_back(alarm);
                    LOG(INFO) << appendtext;
                }
                
                for (int rect_index = 0; rect_index < alarm.rects_size(); ++rect_index) {
                    const AnalysisRect& rect = alarm.rects(rect_index);
                    //cv::rectangle(img, cv::Rect(rect.x(), rect.y(), rect.width(), rect.height()), cv::Scalar(0,255,0));
                }
                puttext.putText(img, appendtext, cv::Point(xpos, ypos), cv::Scalar(255,0,0), 20);
                ypos += 20*1.5;
                break;
        }
    }
#ifdef _DEBUG
    if (config.has_display() && config.display()) {
        std::string window_name = std::to_string(config.video_id()) + std::string("_AppendImg");
if(img.cols>2000)
{
cv::Mat show;
	resize(img, show, cv::Size(img.cols/3,img.rows/3));
        cv::imshow(window_name, show);
        cv::waitKey(10);
}
else
{
        cv::imshow(window_name, img);
        cv::waitKey(10);
}
    }
#endif

    if (alarms.empty())
        return;
    struct timeval time_now;
    gettimeofday(&time_now, NULL);
    uint32_t diff_time = (time_now.tv_sec - last_time.tv_sec)*1000 + (time_now.tv_usec - last_time.tv_usec)/1000;
    if (diff_time < 1000)
        return;
    last_time = time_now;
    for (AnalysisAlarm& info : alarms/*这里的alarm具备所有报警信息*/) {
        AlarmData* this_alarm = send_queue->get_head_to_write();
        if (this_alarm == NULL)
            continue;
        this_alarm->clear();//
        this_alarm->camera_id = config.video_id();
        this_alarm->scene_type = info.scene_type(); 
	char currentTime[64];
	time_t t=time(0);
        t+=8*60*60;
        tm *beijingTime;
        beijingTime=localtime(&(t));
	strftime(currentTime, sizeof(currentTime), "%Y-%m-%d %H:%M:%S",beijingTime);
        this_alarm->current_time = currentTime;
	this_alarm->img = img.clone();

        switch(info.scene_type()) {
        case SEGMANTIC:
            break;
        case WATERCOLOR:
           this_alarm->Color=info.water_color();
            break;
        case WATERGAUGE:
           this_alarm->DraftValue=info.water_gauge_num();
            break;
   case FLOATER:
             this_alarm->IsActive = true;
            this_alarm->TotalArea = info.floater_area();
            this_alarm->Speed = info.floater_speed();
            for (int i = 0; i < info.rects_size(); ++i) {
              AnalysisRect rect = info.rects(i);
              cv::Rect rect_T(rect.x(), rect.y(), rect.width(), rect.height());
              this_alarm->rectangle_array_vect.push_back(rect_T);
             }
            break;

        case INVADE:
            this_alarm->IsActive = true;
         for (int i = 0; i < info.rects_size(); ++i) {
                    AnalysisRect rect = info.rects(i);
                    cv::Rect rect_T(rect.x(), rect.y(), rect.width(), rect.height());
                    this_alarm->rectangle_array_vect.push_back(rect_T);
                }
           
            break;
        case FISHING:
            this_alarm->IsActive = true;
          for (int rect_index = 0; rect_index < info.rects_size(); ++rect_index) {
              AnalysisRect rect = info.rects(rect_index);
              cv::Rect rect_T(rect.x(), rect.y(), rect.width(), rect.height());
              this_alarm->rectangle_array_vect.push_back(rect_T);
                }
            break;
        case LITTER:
            this_alarm->IsActive = true;
          for (int i = 0; i < info.rects_size(); ++i) {
            AnalysisRect rect = info.rects(i);
            cv::Rect rect_T(rect.x(), rect.y(), rect.width(), rect.height());
            this_alarm->rectangle_array_vect.push_back(rect_T);
           }
            break;
        case SWIMING:
            this_alarm->IsActive = true;
            break;
        }
        send_queue->head_next();
    }
}

void Analysis::StartAnalysis() {
    LOG(INFO) << "Analysis Algorithms start";
    send_client->Start();
    for (size_t i = 0; i < algorithms.size(); ++i) {
        RiverThread* inst = algorithms[i];
        inst->Start();
    }
}

void Analysis::StopAnalysis() {
    LOG(INFO) << "Analysis Algorithms stop";
    for (size_t i = 0; i < algorithms.size(); ++i) {
        RiverThread* inst = algorithms[i];
        inst->Stop();
    }
    send_client->Stop();
}

std::string Analysis::GetResult(const cv::Mat& input) {
    SetOriginImg(input);
    return "";
}


