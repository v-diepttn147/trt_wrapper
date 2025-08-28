#include "object_detect/infer_2dod.h"

namespace object_detector_trt {

constexpr int c_bbox_size_bytes = 4;
constexpr int c_log_severity_level = 3;
constexpr float c_max_intensity = 255.0F;
constexpr double c_seconds_to_ms = 1000.0;

Infer2DOD::Infer2DOD(const std::string& model_path) {
    trt_infer_ = new TRTInferAPI(model_path, /*enableFP16=*/true, /*workspaceMB=*/10240);
    inDims  = trt_infer_->getInputDims();
    outDims = trt_infer_->getOutputDims();
}

Infer2DOD::~Infer2DOD() {
    delete trt_infer_;
}

void Infer2DOD::preprocess(const cv::Mat& img, float* gpu_input) {
    
    auto t0 = std::chrono::high_resolution_clock::now();
    const int N = inDims.d[0]; // expect 1
    const int C = inDims.d[1];
    const int H = inDims.d[2];
    const int W = inDims.d[3];
    
    // CPU: BGR -> RGB, resize to (W,H) but KEEP uint8
    cv::Mat rgb, resized;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(W, H), 0, 0, cv::INTER_LINEAR); // CV_8UC3
    if (img.cols >= img.rows) {
        resize_scale_ = static_cast<float>(img.cols) / static_cast<float>(W);
    } else {
        resize_scale_ = static_cast<float>(img.rows) / static_cast<float>(H);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Preprocess CPU time: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
     // Upload uint8 NHWC to device
    const size_t stepBytes = static_cast<size_t>(resized.step);         // bytes per row (handles alignment)
    const size_t bytes     = stepBytes * static_cast<size_t>(resized.rows);
    unsigned char* dNHWC8  = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&dNHWC8), bytes);
    cudaMemcpy(dNHWC8, resized.data, bytes, cudaMemcpyHostToDevice);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Preprocess upload time: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

    // Launch: uint8 NHWC -> fp32 NCHW (normalize inside kernel)
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};

    cudaStream_t stream{};
    cudaStreamCreate(&stream);

    launch_nhwc_to_nchw_norm(
        dNHWC8, H, W, stepBytes,
        gpu_input, /* swapRB = */ false, /* already RGB */
        1.0f/255.0f, mean, stdv, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(dNHWC8);
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Preprocess kernel time: " << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms\n";   
}

void Infer2DOD::postprocess(const float* gpu_output, int output_size, std::vector<DetectionResult>& output) {

    auto t0 = std::chrono::high_resolution_clock::now();
    // TODO: modify according to output format
    std::vector<float> host_output(output_size);
    cudaMemcpy(host_output.data(), gpu_output, sizeof(float) * output_size, cudaMemcpyDeviceToHost);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Postprocess download time: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
    const int num_detections = static_cast<int>(host_output[0]);
    output.clear();
    output.reserve(num_detections);

    // for (int i = 0; i < num_detections; ++i) {
    //     DetectionResult det;
    //     det.class_id = static_cast<int>(host_output[1 + i * 6 + 1]);
    //     det.confidence = host_output[1 + i * 6 + 2];
    //     det.box.x = static_cast<int>(host_output[1 + i * 6 + 3]);
    //     det.box.y = static_cast<int>(host_output[1 + i * 6 + 4]);
    //     det.box.width = static_cast<int>(host_output[1 + i * 6 + 5] - det.box.x);
    //     det.box.height = static_cast<int>(host_output[1 + i * 6 + 6] - det.box.y);
    //     output.push_back(det);
    // }

    // The following code is adapted from adasvf
    const int signal_result_dims = static_cast<int>(outDims.d[1]);
    const int stride_num = static_cast<int>(outDims.d[2]);

    std::vector<int> class_ids{};
    std::vector<float> confidences{};
    std::vector<cv::Rect> boxes{};

    // Note: transpose is needed here since we follow ultralytics conventions
    cv::Mat raw_data = cv::Mat{signal_result_dims, stride_num, CV_32F, host_output.data()}.t();

    int data_index{0};
    for (int i = 0; i < stride_num; ++i) {
        float* class_scores = &raw_data.at<float>(data_index + c_bbox_size_bytes);
        const cv::Mat scores{1, static_cast<int>(classes_.size()), CV_32FC1, class_scores};

        cv::Point class_id{};
        double max_class_score{0};

        // TODO (14068): Opencv dependency will be removed
        cv::minMaxLoc(scores, nullptr, &max_class_score, nullptr, &class_id);
        if (max_class_score > confidence_threshold) {

            confidences.push_back(static_cast<float>(max_class_score));
            class_ids.push_back(class_id.x);

            const float x = raw_data.at<float>(data_index);
            const float y = raw_data.at<float>(data_index + 1);
            const float w = raw_data.at<float>(data_index + 2);
            const float h = raw_data.at<float>(data_index + 3);

            const int left = static_cast<int>((x - 0.5 * w) * resize_scale_);
            const int top = static_cast<int>((y - 0.5 * h) * resize_scale_);
            const int width = static_cast<int>(w * resize_scale_);
            const int height = static_cast<int>(h * resize_scale_);

            boxes.emplace_back(left, top, width, height);
        }
        data_index += signal_result_dims;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Postprocess CPU time: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

    // We need to use std::vec here in order for cv::dnn can be used.
    // TODO (14068): Opencv dependency will be removed
    std::vector<int> nms_results;
    cv::dnn::NMSBoxes(
        boxes, confidences, confidence_threshold, iou_threshold, nms_results);

    output.clear();
    for (const auto& idx: nms_results) {
        output.push_back({class_ids[idx], confidences[idx], 
            {boxes[idx].x, boxes[idx].y, boxes[idx].width, boxes[idx].height}});
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Postprocess NMS time: " << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms\n";
    std::cout << "First output: " << output[0].class_id << " " << output[0].confidence << " " 
              << output[0].box.x << " " << output[0].box.y << " "
              << output[0].box.width << " " << output[0].box.height << "\n";
}

void Infer2DOD::infer(const cv::Mat& input_image, std::vector<DetectionResult>& output) {
    
    // Preprocess
    float* gpu_input = reinterpret_cast<float*>(trt_infer_->getDeviceInputBuffer(0));
    preprocess(input_image, gpu_input);

    // Inference
    trt_infer_->enqueueOn();
    cudaStreamSynchronize(0);

    // print first few sample output
    const float* gpu_output = reinterpret_cast<const float*>(trt_infer_->getDeviceOutputBuffer(0));
    int output_size = 1;  
    for (int i = 0; i < outDims.nbDims; ++i) {
        output_size *= outDims.d[i];
    }
    postprocess(gpu_output, output_size, output); 
    
    std::cout << output[0].class_id << " " << output[0].confidence << " " 
              << output[0].box.x << " " << output[0].box.y << " "
              << output[0].box.width << " " << output[0].box.height << "\n";
    
}

}