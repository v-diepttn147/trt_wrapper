// tests/test_trt_infer.cpp
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <numeric>
#include <random>
#include <chrono>

#include "trt_wrapper/trt_infer_api.h" 

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
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <model.onnx|model.engine> [runs=1]\n";
            return 2;
        }
        const std::string modelPath = argv[1];
        const int runs = (argc >= 3) ? std::max(1, std::stoi(argv[2])) : 1;

        // Build/Load engine. We disable FP16 so inputs remain FP32 for this simple test.
        TRTInferAPI trt(modelPath, /*enableFP16=*/false, /*workspaceMB=*/10240);

        // Query I/O
        const auto inDims  = trt.getInputDims();
        const auto outDims = trt.getOutputDims();

        const size_t inElems  = volume(inDims);
        const size_t outElems = volume(outDims);

        std::cout << "Loaded model: " << modelPath << "\n"
                  << "Input dims : " << dimsToString(inDims)  << " -> " << inElems  << " elements (float32)\n"
                  << "Output dims: " << dimsToString(outDims) << " -> " << outElems << " elements (float32)\n"
                  << "Runs       : " << runs << "\n";

        // Allocate host buffers (float32). If your engine uses a different I/O dtype,
        // adapt accordingly or enable FP16 and pass half-precision buffers.
        std::vector<float> hInput(inElems);
        std::vector<float> hOutput(outElems);

        // Fill input with deterministic pseudo-random values
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> uni(0.f, 1.f);
        for (auto& v : hInput) v = uni(rng);

        // Warm-up
        trt.infer(hInput.data(),  sizeof(float) * hInput.size(),
                  hOutput.data(), sizeof(float) * hOutput.size());

        // Timed runs
        double msTotal = 0.0;
        for (int i = 0; i < runs; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            trt.infer(hInput.data(),  sizeof(float) * hInput.size(),
                      hOutput.data(), sizeof(float) * hOutput.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            msTotal += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        std::cout << "Avg latency: " << (msTotal / runs) << " ms over " << runs << " run(s)\n";

        // Print output shapes

        // Print first few outputs
        const size_t show = std::min<size_t>(std::size_t(10), hOutput.size());
        std::cout << "Output[0:" << show << "): ";
        for (size_t i = 0; i < show; ++i) {
            std::cout << hOutput[i] << (i + 1 < show ? ", " : "\n");
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
