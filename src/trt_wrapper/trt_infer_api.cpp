#include "trt_wrapper/trt_infer_api.h"
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <stdexcept>
#include <cstring>   // std::memcpy
#include <algorithm> // std::max

using namespace nvinfer1;

// void TRTInferAPI::Logger::log(Severity s, const char* msg) noexcept {
//     if (s <= Severity::kWARNING) {
//         fprintf(stderr, "[TRT] %s\n", msg);
//     }
// }

static bool hasSuffix(const std::string& s, const std::string& suf) {
    if (s.size() < suf.size()) return false;
    return std::equal(suf.rbegin(), suf.rend(), s.rbegin());
}

std::vector<char> TRTInferAPI::readFile_(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("Cannot open file: " + path);
    std::streamsize sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<char> buf(sz);
    if (!ifs.read(buf.data(), sz)) throw std::runtime_error("Failed to read: " + path);
    return buf;
}

size_t TRTInferAPI::elementSize_(DataType dt) {
    switch (dt) {
        case DataType::kFLOAT:  return 4;
        case DataType::kHALF:   return 2;
        case DataType::kINT8:   return 1;
        case DataType::kINT32:  return 4;
        case DataType::kBOOL:   return 1;
        case DataType::kFP8:    return 1;
        default: throw std::runtime_error("Unsupported TensorRT DataType");
    }
}

size_t TRTInferAPI::volume_(const Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
    return v;
}

TRTInferAPI::TRTInferAPI(const std::string& modelPath, bool enableFP16, size_t workspaceMB) {
    if (hasSuffix(modelPath, ".onnx"))
        buildFromOnnx_(modelPath, enableFP16, workspaceMB);
    else if (hasSuffix(modelPath, ".trt"))
        loadFromEngine_(modelPath);
    else
        throw std::runtime_error("Unknown model type: " + modelPath);

    if (!engine_) throw std::runtime_error("Engine creation failed");
    context_ = engine_->createExecutionContext();
    if (!context_) throw std::runtime_error("createExecutionContext failed");

    allocBindings_();
}

TRTInferAPI::~TRTInferAPI() {
     for (void* p : deviceBindings_) if (p) cudaFree(p);
  deviceBindings_.clear();

  delete context_;  context_ = nullptr;
  delete engine_;   engine_  = nullptr;

  // If you still keep build-time objects as members:
  delete network_;  network_ = nullptr;
  delete config_;   config_  = nullptr;
  delete builder_;  builder_ = nullptr;
  delete runtime_;  runtime_ = nullptr;
  delete static_cast<nvonnxparser::IParser*>(parser_); parser_ = nullptr;
}

void TRTInferAPI::buildFromOnnx_(const std::string& onnxPath, bool fp16, size_t workspaceMB) {
    // Explicit-batch network
    // uint32_t flags = 1u << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    builder_ = createInferBuilder(logger_);
    if (!builder_) throw std::runtime_error("createInferBuilder failed");

    network_ = builder_->createNetworkV2(0U);
    if (!network_) throw std::runtime_error("createNetworkV2 failed");

    // ONNX parser
    auto* parser = nvonnxparser::createParser(*network_, logger_);
    if (!parser) throw std::runtime_error("createParser failed");
    parser_ = parser;

    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(ILogger::Severity::kINTERNAL_ERROR))) {
        throw std::runtime_error("Failed to parse ONNX: " + onnxPath);
    }

    config_ = builder_->createBuilderConfig();
    if (!config_) throw std::runtime_error("createBuilderConfig failed");

    // Workspace
    config_->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, workspaceMB * (1ULL << 20));
    config_->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);

    // FP16 if hardware supports
    if (fp16 && builder_->platformHasFastFp16())
        config_->setFlag(BuilderFlag::kFP16);

    // Build engine
    IHostMemory* plan = builder_->buildSerializedNetwork(*network_, *config_);
    if (!plan) throw std::runtime_error("buildSerializedNetwork failed");

    runtime_ = createInferRuntime(logger_);
    if (!runtime_) throw std::runtime_error("createInferRuntime failed");

    engine_ = runtime_->deserializeCudaEngine(plan->data(), plan->size());
    if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");

    // --- serialize & save engine next to the ONNX ---
    // Save plan next to ONNX as .trt
    {
        IHostMemory* blob = engine_->serialize();
        if (!blob) { delete plan; throw std::runtime_error("serialize() returned null"); }
        std::string enginePath = onnxPath;
        if (auto dot = enginePath.find_last_of('.'); dot != std::string::npos) enginePath.erase(dot);
        enginePath += ".trt";

        std::ofstream ofs(enginePath, std::ios::binary);
        if (!ofs) { delete blob; delete plan; throw std::runtime_error("Cannot open for write: " + enginePath); }
        ofs.write(static_cast<const char*>(blob->data()), static_cast<std::streamsize>(blob->size()));
        bool ok = ofs.good(); ofs.close();
        delete blob;
        if (!ok) { delete plan; throw std::runtime_error("Failed to write engine: " + enginePath); }
    }
    // -----------------------------------------------------
}

void TRTInferAPI::loadFromEngine_(const std::string& enginePath) {
    auto blob = readFile_(enginePath);
    runtime_ = createInferRuntime(logger_);
    if (!runtime_) throw std::runtime_error("createInferRuntime failed");

    engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
    if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");
}

void TRTInferAPI::allocBindings_() {
    const int n = engine_->getNbIOTensors();
    if (n < 2) throw std::runtime_error("Expected at least 1 input and 1 output tensor");

    ioNames_.clear(); ioNames_.reserve(n);
    deviceBindings_.assign(n, nullptr);
    inputIndex_ = outputIndex_ = -1;

    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);
        ioNames_.emplace_back(name);

        const auto mode  = engine_->getTensorIOMode(name);      // kINPUT / kOUTPUT
        const auto dtype = engine_->getTensorDataType(name);
        // For static shapes, engine shape is final; for dynamic you would use context->getTensorShape(name)
        const auto shape = engine_->getTensorShape(name);

        // This wrapper assumes static shapes:
        size_t bytes = volume_(shape) * elementSize_(dtype);

        void* dev = nullptr;
        if (cudaMalloc(&dev, bytes) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed for tensor: " + std::string(name));
        }
        deviceBindings_[i] = dev;

        // Bind device address by NAME (this replaces the old bindings array)
        if (!context_->setTensorAddress(name, dev))
            throw std::runtime_error("setTensorAddress failed for: " + std::string(name));

        if (mode == nvinfer1::TensorIOMode::kINPUT  && inputIndex_  < 0) inputIndex_  = i;
        if (mode == nvinfer1::TensorIOMode::kOUTPUT && outputIndex_ < 0) outputIndex_ = i;
    }

    if (inputIndex_ < 0 || outputIndex_ < 0)
        throw std::runtime_error("Could not identify input/output tensors");
}

void* TRTInferAPI::getDeviceInputBuffer(int idx) const {
    int count = 0;
    const int n = engine_->getNbIOTensors();
    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            if (count == idx) return deviceBindings_[i];
            ++count;
        }
    }
    return nullptr; // idx out of range
}
void* TRTInferAPI::getDeviceOutputBuffer(int idx) const {
    int count = 0;
    const int n = engine_->getNbIOTensors();
    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            if (count == idx) return deviceBindings_[i];
            ++count;
        }
    }
    return nullptr; // idx out of range
}

bool TRTInferAPI::enqueueOn(cudaStream_t stream) {
    // If no stream passed, create a short-lived one
    if (!stream) {
        cudaStream_t s{};
        if (cudaStreamCreate(&s) != cudaSuccess) return false;
        bool ok = context_->enqueueV3(s);
        cudaStreamSynchronize(s);
        cudaStreamDestroy(s);
        return ok;
    }
    // return context_->enqueueV2(deviceBindings_.data(), stream, nullptr);
    return context_->enqueueV3(stream);
}

void TRTInferAPI::infer(const void* hostInput, size_t inputBytes,
                        void* hostOutput, size_t outputBytes)
{
    // Name-based shapes/types (static)
    const char* inName  = ioNames_[inputIndex_].c_str();
    const char* outName = ioNames_[outputIndex_].c_str();

    Dims inD   = engine_->getTensorShape(inName);
    Dims outD  = engine_->getTensorShape(outName);
    DataType inT  = engine_->getTensorDataType(inName);
    DataType outT = engine_->getTensorDataType(outName);

    size_t needIn  = volume_(inD)  * elementSize_(inT);
    size_t needOut = volume_(outD) * elementSize_(outT);
    if (needIn != inputBytes)
        throw std::runtime_error("Input size mismatch: got " + std::to_string(inputBytes) +
                                 ", expected " + std::to_string(needIn));
    if (needOut != outputBytes)
        throw std::runtime_error("Output size mismatch: got " + std::to_string(outputBytes) +
                                 ", expected " + std::to_string(needOut));

    void* dIn  = deviceBindings_[inputIndex_];
    void* dOut = deviceBindings_[outputIndex_];

    cudaStream_t stream{};
    if (cudaStreamCreate(&stream) != cudaSuccess)
        throw std::runtime_error("cudaStreamCreate failed");

    if (cudaMemcpyAsync(dIn, hostInput, inputBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        cudaStreamDestroy(stream);
        throw std::runtime_error("H2D memcpy failed");
    }

    if (!context_->enqueueV3(stream)) {
        cudaStreamDestroy(stream);
        throw std::runtime_error("enqueueV3 failed");
    }

    if (cudaMemcpyAsync(hostOutput, dOut, outputBytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        cudaStreamDestroy(stream);
        throw std::runtime_error("D2H memcpy failed");
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

nvinfer1::Dims TRTInferAPI::getInputDims(int idx) const {
    // Return the idx-th INPUT tensor shape in engine I/O order (static)
    int count = 0;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            if (count == idx) return engine_->getTensorShape(name);
            ++count;
        }
    }
    throw std::out_of_range("Input index out of range");
}

nvinfer1::Dims TRTInferAPI::getOutputDims(int idx) const {
    int count = 0;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == TensorIOMode::kOUTPUT) {
            if (count == idx) return engine_->getTensorShape(name);
            ++count;
        }
    }
    throw std::out_of_range("Output index out of range");
}
