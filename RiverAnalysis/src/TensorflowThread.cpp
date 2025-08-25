#include "TensorflowThread.h"
#include "Analysis.h"

#include "DefineColor.h"

#include <iostream>
#include <vector>

#include <sys/time.h>
#include <glog/logging.h>

using namespace tensorflow;

Session* TensorflowThread::session = NULL;

const int IMG_SIZE = 513;
const char* input_label = "ImageTensor:0";
const char* output_label = "SemanticPredictions:0";

TensorflowThread::TensorflowThread(Analysis* analysis_manager)
    : manager(analysis_manager), 
    input_tensor(DT_UINT8, TensorShape({1, IMG_SIZE, IMG_SIZE, 3})) {
        
}

int TensorflowThread::Initalize(const char* model_path) {
    SessionOptions options;
    ConfigProto& config = options.config;
    config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.7);
    if (session != NULL || model_path == NULL) {
        std::cout << "Error Input" << std::endl;
        return -1;
    }
    Status status = NewSession(options, &session);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return -1;
    }
    
    GraphDef graphdef;
    status = ReadBinaryProto(Env::Default(), model_path, &graphdef);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return -1;
    }
    
    status = session->Create(graphdef);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return -1;
    }

    Tensor init_tensor(DT_UINT8, TensorShape({1, IMG_SIZE, IMG_SIZE, 3}));
    uint8_t* tensor_ptr = init_tensor.flat<uint8_t>().data();
    memset(tensor_ptr, 0, IMG_SIZE * IMG_SIZE * 3);
    
    std::vector<std::pair<std::string, Tensor> > inputs = { {input_label, init_tensor} };
    std::vector<Tensor> outputs;
    status = session->Run(inputs, {output_label}, {}, &outputs);
    if (!status.ok() || outputs.empty()) {
        std::cout << status.ToString() << std::endl;
        return -1;
    }
    return 0;
}

void TensorflowThread::Uninitalize() {
   if (session == NULL)
       return;
   session->Close();
   delete session;
   session = NULL;
}

void TensorflowThread::Run() {
    std::lock_guard<std::mutex> lk(_mutex);
    LOG(INFO) << "TensorflowThread start";
    _is_run = true;
    uint32_t interval = config.detect_interval() * 1000;
    struct timeval start, end;
    while (_is_run) {
        gettimeofday(&start, NULL);
        cv::Mat origin_img = manager->GetOriginImg();
        if (origin_img.empty()) {
            usleep(interval);
            continue;
        }
        
        uint8_t* img_ptr = input_tensor.flat<uint8_t>().data();
        cv::Mat convertImg(IMG_SIZE, IMG_SIZE, CV_8UC3, img_ptr);
        cv::resize(origin_img, convertImg, cv::Size(IMG_SIZE, IMG_SIZE));

        std::vector<std::pair<std::string, Tensor> > inputs = { {input_label, input_tensor} };
        std::vector<Tensor> outputs;
        Status status = session->Run(inputs, {output_label}, {}, &outputs);
        if (!status.ok() || outputs.empty()) {
            std::cout << status.ToString() << std::endl;
            usleep(interval);
            continue;
        }

        StringPiece output_matrix = outputs[0].tensor_data();

        cv::Mat save_img = cv::Mat::zeros(IMG_SIZE, IMG_SIZE, CV_8UC3);
        uint64_t* ptr = (uint64_t*)(output_matrix.data());

        for (int row = 0; row < IMG_SIZE; ++row) {
            uint64_t* ptr_line = ptr + IMG_SIZE * row;
            for (int col = 0; col < IMG_SIZE; ++col) {
                int num = ptr_line[col];
                const cv::Scalar& thiscolor = get_color_of_label(num);
                save_img.at<cv::Vec3b>(row, col)[0] = thiscolor[0];
                save_img.at<cv::Vec3b>(row, col)[1] = thiscolor[1];
                save_img.at<cv::Vec3b>(row, col)[2] = thiscolor[2];
            }
        }
#ifdef _DEBUG
        if (config.has_display() && config.display()) {
            std::string window_name = std::to_string(manager->Config().video_id());
            std::string origin_name = window_name + std::string("_Segment_origin");
            std::string segment_name = window_name + std::string("_Segment_sement");
            cv::imshow(origin_name, origin_img);
            cv::imshow(segment_name, save_img);
            cv::waitKey(10);
        }
#endif
        manager->SetSegmentImg(save_img);
        gettimeofday(&end, NULL);
        std::cout << (end.tv_sec - start.tv_sec)*1000 + (end.tv_usec - start.tv_usec)/1000 << std::endl;
        usleep(interval);
    }
    LOG(INFO) << "TensorflowThread end";
    _is_run = false;
    CallStop();
}

