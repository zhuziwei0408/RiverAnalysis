#ifndef RIVER_ANALYSIS_MANAGER_H
#define RIVER_ANALYSIS_MANAGER_H

#include "Analysis.h"

class AnalysisManager {
public:
    AnalysisManager();
    ~AnalysisManager();

    static int Initalize(const char* argv0, const char* model_path);
    static void Uninitalize();

    int run();
    int LoadConfig(const char* argv, const char* config_path);

private:
    static std::once_flag init_flag;
    static std::once_flag uninit_flag;
    static int init_result;

    std::vector<Analysis*> inst_vec;//分析实例
    ConfigList config;
};

#endif
