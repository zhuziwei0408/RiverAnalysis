#ifndef RIVER_ANALYSIS_TENSORFLOW_THREAD_H
#define RIVER_ANALYSIS_TENSORFLOW_THREAD_H

#include "RiverThread.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

class Analysis;
/*功能：加载训练模型，完成视频分割，将分割图保存到manager->SetSegmentImg中*/
class TensorflowThread
    : public RiverThread {
public:
    TensorflowThread(Analysis* analysis_manager);
    ~TensorflowThread() {}

    void Run();

public:
    static int Initalize(const char* model_path);
    static void Uninitalize();
private:
    static tensorflow::Session* session;
    Analysis* manager;
    tensorflow::Tensor input_tensor;
};

#endif
