#pragma once
#include <string>
#include <vector>
#include <NvInfer.h>

class TRTInferAPI {
public:
    explicit TRTInferAPI(const std::string& modelPath,
                         bool enableFP16 = true,
                         size_t workspaceMB = 1024);
    ~TRTInferAPI();

    // Basic single-input/single-output API.
    // hostInput/hostOutput are host pointers; sizes are in bytes.
    // For static shapes only (no dynamic dimensions).
    void infer(const void* hostInput, size_t inputBytes,
               void* hostOutput, size_t outputBytes);

    nvinfer1::Dims getInputDims(int idx = 0) const;
    nvinfer1::Dims getOutputDims(int idx = 0) const;
    void* getDeviceInputBuffer(int idx = 0)  const;
    void* getDeviceOutputBuffer(int idx = 0) const;
    bool enqueueOn(cudaStream_t stream = nullptr);
private:
    // Build/Load
    void buildFromOnnx_(const std::string& onnxPath, bool fp16, size_t workspaceMB);
    void loadFromEngine_(const std::string& enginePath);
    static std::vector<char> readFile_(const std::string& path);

    // Buffers
    void allocBindings_();
    static size_t elementSize_(nvinfer1::DataType dt);
    static size_t volume_(const nvinfer1::Dims& d);

    // TRT objects
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity s, const char* msg) noexcept override;
    } logger_;

    nvinfer1::IRuntime*           runtime_  = nullptr;
    nvinfer1::ICudaEngine*        engine_   = nullptr;
    nvinfer1::IExecutionContext*  context_  = nullptr;
    nvinfer1::IBuilder*           builder_  = nullptr;
    nvinfer1::INetworkDefinition* network_  = nullptr;
    nvinfer1::IBuilderConfig*     config_   = nullptr;
    void*                         parser_   = nullptr; // forward-declare to avoid header include

    // Device buffers for bindings (simple case: 1 in, 1 out)
    std::vector<void*> deviceBindings_;
    int inputIndex_  = -1;
    int outputIndex_ = -1;
};
