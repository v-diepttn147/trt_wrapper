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