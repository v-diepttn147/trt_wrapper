#include "object_detect/infer_2dod.h"

namespace object_detector_trt {

Infer2DOD::Infer2DOD(const std::string& model_path) {
    trt_infer_ = new TRTInferAPI(model_path, /*enableFP16=*/true, /*workspaceMB=*/10240);
    inDims  = trt_infer_->getInputDims();
    outDims = trt_infer_->getOutputDims();
}

Infer2DOD::~Infer2DOD() {
    delete trt_infer_;
}

void Infer2DOD::preprocess(const cv::Mat& img, float* gpu_input) {
    const int N = inDims.d[0]; // expect 1
    const int C = inDims.d[1];
    const int H = inDims.d[2];
    const int W = inDims.d[3];
    
    // CPU: BGR -> RGB, resize to (W,H) but KEEP uint8
    cv::Mat rgb, resized;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(W, H), 0, 0, cv::INTER_LINEAR); // CV_8UC3

     // Upload uint8 NHWC to device
    const size_t stepBytes = static_cast<size_t>(resized.step);         // bytes per row (handles alignment)
    const size_t bytes     = stepBytes * static_cast<size_t>(resized.rows);
    unsigned char* dNHWC8  = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&dNHWC8), bytes);
    cudaMemcpy(dNHWC8, resized.data, bytes, cudaMemcpyHostToDevice);

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
}

void Infer2DOD::postprocess(const float* gpu_output, int output_size, std::vector<DetectionResult>& output) {

    // TODO: modify according to output format
    std::vector<float> host_output(output_size);
    cudaMemcpy(host_output.data(), gpu_output, sizeof(float) * output_size, cudaMemcpyDeviceToHost);

    const int num_detections = static_cast<int>(host_output[0]);
    output.clear();
    output.reserve(num_detections);

    for (int i = 0; i < num_detections; ++i) {
        DetectionResult det;
        det.class_id = static_cast<int>(host_output[1 + i * 6 + 1]);
        det.confidence = host_output[1 + i * 6 + 2];
        det.box.x = static_cast<int>(host_output[1 + i * 6 + 3]);
        det.box.y = static_cast<int>(host_output[1 + i * 6 + 4]);
        det.box.width = static_cast<int>(host_output[1 + i * 6 + 5] - det.box.x);
        det.box.height = static_cast<int>(host_output[1 + i * 6 + 6] - det.box.y);
        output.push_back(det);
    }
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