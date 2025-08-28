# Prerequisites

- CuDNN (8.9)
- OpenCV
- TensorRT (10.4)
- cuda

# Run

```bash 
./build/test_2dod <link-to-model-onnx/trt> <num of runs>
```

**Model trt will be generated when running onnx model, use trt model for the next run**

# Performance

1. 2DOD

```
Preprocess CPU time: 0.312266 ms
Preprocess upload time: 0.06563 ms
Preprocess kernel time: 0.12652 ms
23 31.7867 35 37 21 21
Avg latency: 4.15018 ms over 20 run(s)
```