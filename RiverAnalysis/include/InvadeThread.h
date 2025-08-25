#ifndef RIVER_ANALYSIS_INVADE_THREAD_H
#define RIVER_ANALYSIS_INVADE_THREAD_H

#include "RiverThread.h"
#include "opencv2/opencv.hpp"
class Analysis;

class InvadeThread : public RiverThread {
public:
    InvadeThread(Analysis* analysis_manager);
    ~InvadeThread() {}

    void Run();
    int LoadConfig(const AlgorithmConfig& input_config);

    AnalysisAlarm GetAlarm();
    void SetAlarm(bool is_active, const std::vector<cv::Rect>& result);
private:
    Analysis* manager;
};

#endif
