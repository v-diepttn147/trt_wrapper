#pragma once
#include <string>
#include "trt_wrapper/trt_infer_api.h" 
#include "trt_wrapper/preprocess_cuda.h"
#include "opencv2/opencv.hpp"

namespace object_detector_trt {
struct BoundingBox2D {
    // x, y: top-left corner in image coordinates
    // following OpenCV convention
    int x{0};
    int y{0};
    int width{0};
    int height{0};
};

struct DetectionResult {
    int class_id{-1};
    float confidence{0.0F};
    BoundingBox2D box{};
};

class Infer2DOD {
public:
    Infer2DOD(const std::string& model_path);
    ~Infer2DOD();
    void infer(const cv::Mat& input_image, std::vector<DetectionResult>& output);
private:
    // Add private members as needed
    void preprocess(const cv::Mat& img, float* gpu_input);
    void postprocess(const float* gpu_output, int output_size, std::vector<DetectionResult>& output);

    TRTInferAPI* trt_infer_;
    nvinfer1::Dims inDims;
    nvinfer1::Dims outDims;
    std::vector<std::string> classes_{{"car", "person", "motorcycle"}};
    float confidence_threshold = 0.1f;
    float iou_threshold = 0.1f;
    float resize_scale_;
};
}
