#ifndef RIVER_ANALYSIS_THREAD_H
#define RIVER_ANALYSIS_THREAD_H

#include <thread>
#include <mutex>
#include <condition_variable>

#include "AnalysisConfig.pb.h"

class RiverThread {
public:
    RiverThread();
    virtual ~RiverThread() {}

    void Start();
    void Stop();
    
    virtual void Run() = 0;
    virtual AnalysisAlarm GetAlarm() {
        return AnalysisAlarm();
    };
    //每一种算法的config
    virtual int LoadConfig(const AlgorithmConfig& input_config) {
        config = input_config;
        alarm.set_scene_type(config.algorithm_type());
        return 0;
    }
    
    void WaitFor(uint32_t ms);
    void CallStop();
    void WaitUtilDie();

    volatile bool _is_run;
    AlgorithmConfig config;
 
protected:
    std::thread* _thread;
    std::mutex _mutex;
    std::condition_variable _wait_stop_cond;

    AnalysisAlarm alarm;
    std::mutex alarm_mutex;
};

#endif
