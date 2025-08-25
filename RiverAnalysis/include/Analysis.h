#ifndef RIVER_ANALYSIS_H
#define RIVER_ANALYSIS_H

#include "RiverThread.h"
#include "opencv2/opencv.hpp"

#include "cv320PutChText.h"
#include "AnalysisConfig.pb.h"
#include "http_client.h"
#include "GaussBgModeling.h"

class Analysis : public RiverThread {
public:
    Analysis();
    ~Analysis();

    void Run();

    std::string GetResult(const cv::Mat& input);
    int LoadConfig(const char* config_path);

    void StartAnalysis();
    void StopAnalysis();

    RiverThread* GetAlgorithm(const AlgorithmConfig& config);

public:
    void SetOriginImg(const cv::Mat& input) {
        // std::cout << "lock" << std::endl;
        std::lock_guard<std::mutex> lk(origin_mutex);
        input.copyTo(origin_img);
    }
    cv::Mat GetOriginImg() {
        // std::cout << "unlock" << std::endl;
        std::lock_guard<std::mutex> lk(origin_mutex);
        cv::Mat res = origin_img.clone();
        return res;
    }

    void SetSegmentImg(const cv::Mat& input) {
        // std::cout << "slock" << std::endl;
        std::lock_guard<std::mutex> lk(segment_mutex);
        input.copyTo(segment_img);
    }
    cv::Mat GetSegmentImg() {
        // std::cout << "sunlock" << std::endl;
        std::lock_guard<std::mutex> lk(segment_mutex);
        cv::Mat res = segment_img.clone();
        return res;
    }

    void SetForegroundImg(const cv::Mat& input) {
        // std::cout << "slock" << std::endl;
        std::lock_guard<std::mutex> lk(foreground_mutex);
        input.copyTo(foreground_img);
    }
    cv::Mat GetForegroundImg() {
        // std::cout << "sunlock" << std::endl;
        std::lock_guard<std::mutex> lk(foreground_mutex);
        cv::Mat res = foreground_img.clone();
        return res;
    }

public:
    const AnalysisConfig& Config() const {
        return config;
    }

private:
    void DrawInfo(cv::Mat& img,cv::Mat segmentImg);
private:
    std::vector<RiverThread*> algorithms;
    
    AnalysisConfig config;
    HttpClient* send_client;
    AlarmMsgQueue* send_queue;
    GaussBgModeling *gauss_bg_modeling;

    struct timeval last_time;

    cv320PutChText puttext;

    cv::Mat origin_img;
    std::mutex origin_mutex;

    cv::Mat segment_img;
    std::mutex segment_mutex;

    
    cv::Mat foreground_img;
    std::mutex foreground_mutex;
    cv::Ptr<cv::BackgroundSubtractorMOG2> _mod;
};

#endif

