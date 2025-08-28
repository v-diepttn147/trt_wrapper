// tests/test_trt_infer.cpp
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <numeric>
#include <random>
#include <chrono>

#include "trt_wrapper/trt_infer_api.h" 
#include "object_detect/infer_2dod.h"

static size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
    return v;
}

static std::string dimsToString(const nvinfer1::Dims& d) {
    std::string s = "[";
    for (int i = 0; i < d.nbDims; ++i) {
        s += std::to_string(d.d[i]);
        if (i + 1 < d.nbDims) s += "x";
    }
    s += "]";
    return s;
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <model.onnx|model.trt> <img_path> [runs=1]\n";
            return 2;
        }
        const std::string modelPath = argv[1];
        const std::string imgPath = argv[2];
        const int runs = (argc >= 4) ? std::max(1, std::stoi(argv[3])) : 1;
        std::vector<std::string> classes_{{"car", "person", "motorcycle"}};

        // // Build/Load engine. We disable FP16 so inputs remain FP32 for this simple test.
        // TRTInferAPI trt(modelPath, /*enableFP16=*/false, /*workspaceMB=*/10240);

        // // Query I/O
        // const auto inDims  = trt.getInputDims();
        // const auto outDims = trt.getOutputDims();

        // const size_t inElems  = volume(inDims);
        // const size_t outElems = volume(outDims);

        // std::cout << "Loaded model: " << modelPath << "\n"
        //           << "Input dims  : " << dimsToString(inDims)  << " -> " << inElems  << " elements (float32)\n"
        //           << "Output dims : " << dimsToString(outDims) << " -> " << outElems << " elements (float32)\n"
        //           << "Runs        : " << runs << "\n";

        // // Allocate host buffers (float32). If your engine uses a different I/O dtype,
        // // adapt accordingly or enable FP16 and pass half-precision buffers.
        // std::vector<float> hInput(inElems);
        // std::vector<float> hOutput(outElems);

        // // Fill input with deterministic pseudo-random values
        // std::mt19937 rng(1234);
        // std::uniform_real_distribution<float> uni(0.f, 1.f);
        // for (auto& v : hInput) v = uni(rng);

        // // Warm-up
        // trt.infer(hInput.data(),  sizeof(float) * hInput.size(),
        //           hOutput.data(), sizeof(float) * hOutput.size());

        // // Timed runs
        // double msTotal = 0.0;
        // for (int i = 0; i < runs; ++i) {
        //     auto t0 = std::chrono::high_resolution_clock::now();
        //     trt.infer(hInput.data(),  sizeof(float) * hInput.size(),
        //               hOutput.data(), sizeof(float) * hOutput.size());
        //     auto t1 = std::chrono::high_resolution_clock::now();
        //     msTotal += std::chrono::duration<double, std::milli>(t1 - t0).count();
        // }
        // std::cout << "Avg latency: " << (msTotal / runs) << " ms over " << runs << " run(s)\n";

        // // Print output shapes

        // // Print first few outputs
        // const size_t show = std::min<size_t>(std::size_t(10), hOutput.size());
        // std::cout << "Output[0:" << show << "): ";
        // for (size_t i = 0; i < show; ++i) {
        //     std::cout << hOutput[i] << (i + 1 < show ? ", " : "\n");
        // }

        object_detector_trt::Infer2DOD detector(modelPath);
        std::vector<object_detector_trt::DetectionResult> results;
        cv::Mat img = cv::imread(imgPath);
        if (img.empty()) {
            std::cerr << "Failed to read image\n";
            return 3;
        }
        double msTotal = 0.0;
        for (int i = 0; i < runs; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            detector.infer(img, results);
            auto t1 = std::chrono::high_resolution_clock::now();
            msTotal += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        std::cout << "Avg latency: " << (msTotal / runs) << " ms over " << runs << " run(s)\n";
        // auto t0 = std::chrono::high_resolution_clock::now();
        // detector.infer(img, results);
        // auto t1 = std::chrono::high_resolution_clock::now();
        // std::cout << "Avg latency: " <<  std::chrono::duration<double, std::milli>(t1 - t0).count() << std::endl;

        // visualize results
        for (const auto& r : results) {
            cv::rectangle(img, 
                          cv::Rect(r.box.x, r.box.y, r.box.width, r.box.height), 
                          cv::Scalar(0, 255, 0), 2);
            const std::string label = classes_[r.class_id] + ": " + std::to_string(r.confidence);
            int baseline = 0;
            const auto label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(img, 
                          cv::Point(r.box.x, r.box.y - label_size.height - baseline),
                          cv::Point(r.box.x + label_size.width, r.box.y),
                          cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(img, label, cv::Point(r.box.x, r.box.y - baseline), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
        cv::imwrite("result.png", img);


        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
