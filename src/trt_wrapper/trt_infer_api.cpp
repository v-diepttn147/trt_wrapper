#include "trt_wrapper/trt_infer_api.h"
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <stdexcept>
#include <cstring>   // std::memcpy
#include <algorithm> // std::max

using namespace nvinfer1;

void TRTInferAPI::Logger::log(Severity s, const char* msg) noexcept {
    if (s <= Severity::kWARNING) {
        fprintf(stderr, "[TRT] %s\n", msg);
    }
}

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
    // Free device buffers
    for (void* ptr : deviceBindings_) {
        if (ptr) cudaFree(ptr);
    }
    deviceBindings_.clear();

    if (context_)  context_->destroy();
    if (engine_)   engine_->destroy();
    if (network_)  network_->destroy();
    if (config_)   config_->destroy();
    if (builder_)  builder_->destroy();
    if (runtime_)  runtime_->destroy();

    // parser_ is nvonnxparser::IParser*
    if (parser_)   static_cast<nvonnxparser::IParser*>(parser_)->destroy();
}

void TRTInferAPI::buildFromOnnx_(const std::string& onnxPath, bool fp16, size_t workspaceMB) {
    // Explicit-batch network
    uint32_t flags = 1u << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    builder_ = createInferBuilder(logger_);
    if (!builder_) throw std::runtime_error("createInferBuilder failed");

    network_ = builder_->createNetworkV2(flags);
    if (!network_) throw std::runtime_error("createNetworkV2 failed");

    config_ = builder_->createBuilderConfig();
    if (!config_) throw std::runtime_error("createBuilderConfig failed");

    // Workspace
    config_->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, workspaceMB * (1ULL << 20));

    // FP16 if hardware supports
    if (fp16 && builder_->platformHasFastFp16())
        config_->setFlag(BuilderFlag::kFP16);

    // ONNX parser
    auto* parser = nvonnxparser::createParser(*network_, logger_);
    if (!parser) throw std::runtime_error("createParser failed");
    parser_ = parser;

    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        throw std::runtime_error("Failed to parse ONNX: " + onnxPath);
    }

    // Build engine
    engine_ = builder_->buildEngineWithConfig(*network_, *config_);
    if (!engine_) throw std::runtime_error("buildEngineWithConfig failed");

    // --- serialize & save engine next to the ONNX ---
    nvinfer1::IHostMemory* blob = engine_->serialize();
    if (!blob) throw std::runtime_error("serialize() returned null");

    // Derive "xxx.trt" from "xxx.onnx"
    std::string enginePath;
    const auto dot = onnxPath.find_last_of('.');
    if (dot == std::string::npos) enginePath = onnxPath + ".trt";
    else                           enginePath = onnxPath.substr(0, dot) + ".trt";

    std::ofstream ofs(enginePath, std::ios::binary);
    if (!ofs) {
        blob->destroy();
        throw std::runtime_error("Cannot open for write: " + enginePath);
    }
    ofs.write(static_cast<const char*>(blob->data()),
              static_cast<std::streamsize>(blob->size()));
    const bool ok = ofs.good();
    ofs.close();
    blob->destroy();

    if (!ok) throw std::runtime_error("Failed to write engine to: " + enginePath);
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
    int nb = engine_->getNbBindings();
    if (nb < 2) throw std::runtime_error("Expected at least 1 input and 1 output binding");

    deviceBindings_.resize(nb, nullptr);
    for (int i = 0; i < nb; ++i) {
        bool isInput = engine_->bindingIsInput(i);
        if (isInput && inputIndex_ < 0) inputIndex_ = i;
        if (!isInput && outputIndex_ < 0) outputIndex_ = i;

        Dims d = engine_->getBindingDimensions(i);
        if (std::any_of(d.d, d.d + d.nbDims, [](int x){ return x <= 0; })) {
            throw std::runtime_error("Dynamic or invalid shape detected; this simple wrapper assumes static shapes.");
        }
        DataType dt = engine_->getBindingDataType(i);
        size_t bytes = volume_(d) * elementSize_(dt);
        void* dev = nullptr;
        if (cudaMalloc(&dev, bytes) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed for binding " + std::to_string(i));
        }
        deviceBindings_[i] = dev;
    }
    if (inputIndex_ < 0 || outputIndex_ < 0)
        throw std::runtime_error("Could not identify input/output bindings");
}

void TRTInferAPI::infer(const void* hostInput, size_t inputBytes,
                        void* hostOutput, size_t outputBytes)
{
    // Sanity: check sizes against engine expectations
    {
        Dims inD  = engine_->getBindingDimensions(inputIndex_);
        Dims outD = engine_->getBindingDimensions(outputIndex_);
        DataType inT  = engine_->getBindingDataType(inputIndex_);
        DataType outT = engine_->getBindingDataType(outputIndex_);
        size_t needIn  = volume_(inD)  * elementSize_(inT);
        size_t needOut = volume_(outD) * elementSize_(outT);
        if (needIn != inputBytes)
            throw std::runtime_error("Input size mismatch: got " + std::to_string(inputBytes) +
                                     ", expected " + std::to_string(needIn));
        if (needOut != outputBytes)
            throw std::runtime_error("Output size mismatch: got " + std::to_string(outputBytes) +
                                     ", expected " + std::to_string(needOut));
    }

    cudaStream_t stream{};
    if (cudaStreamCreate(&stream) != cudaSuccess)
        throw std::runtime_error("cudaStreamCreate failed");

    // H2D
    if (cudaMemcpyAsync(deviceBindings_[inputIndex_], hostInput, inputBytes,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        cudaStreamDestroy(stream);
        throw std::runtime_error("H2D memcpy failed");
    }

    // Execute
    if (!context_->enqueueV2(deviceBindings_.data(), stream, nullptr)) {
        cudaStreamDestroy(stream);
        throw std::runtime_error("enqueueV2 failed");
    }

    // D2H
    if (cudaMemcpyAsync(hostOutput, deviceBindings_[outputIndex_], outputBytes,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        cudaStreamDestroy(stream);
        throw std::runtime_error("D2H memcpy failed");
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

nvinfer1::Dims TRTInferAPI::getInputDims(int idx) const {
    int count = 0;
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        if (engine_->bindingIsInput(i)) {
            if (count == idx) return engine_->getBindingDimensions(i);
            ++count;
        }
    }
    throw std::out_of_range("Input index out of range");
}

nvinfer1::Dims TRTInferAPI::getOutputDims(int idx) const {
    int count = 0;
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        if (!engine_->bindingIsInput(i)) {
            if (count == idx) return engine_->getBindingDimensions(i);
            ++count;
        }
    }
    throw std::out_of_range("Output index out of range");
}
