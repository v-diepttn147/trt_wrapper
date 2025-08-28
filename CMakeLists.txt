cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-10)
set(CMAKE_CXX_COMPILER /usr/bin/g++-10)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)

project(integrate_trt LANGUAGES CXX CUDA)

# Global defaults

# set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
# CUDA runtime (cudart) — required for cudaMalloc/MemcpyAsync/Streams
find_package(CUDAToolkit REQUIRED)

# ---------- TensorRT (prebuilt in 3rdparty) ----------
set(TENSORRT_ROOT         ${CMAKE_SOURCE_DIR}/3rdparty/TensorRT-10.4.0.26)
set(TENSORRT_INCLUDE_DIR  ${TENSORRT_ROOT}/include)
set(TENSORRT_LIBRARY_DIR  ${TENSORRT_ROOT}/lib)

# Core libs: add nvinfer (you missed this), plugins, parsers
# set(TENSORRT_LIBRARIES
#     ${TENSORRT_LIBRARY_DIR}/libnvinfer.so
#     ${TENSORRT_LIBRARY_DIR}/libnvinfer_plugin.so
#     ${TENSORRT_LIBRARY_DIR}/libnvinfer_vc_plugin.so
#     ${TENSORRT_LIBRARY_DIR}/libnvonnxparser.so
#     ${TENSORRT_LIBRARY_DIR}/libnvparsers.so
# )
set(TENSORRT_LIBRARIES
    # ${TENSORRT_LIBRARY_DIR}/libnvinfer_dispatch.so
    ${TENSORRT_LIBRARY_DIR}/libnvinfer_plugin.so
    # ${TENSORRT_LIBRARY_DIR}/libnvinfer_lean.so
    ${TENSORRT_LIBRARY_DIR}/libnvinfer_vc_plugin.so
    ${TENSORRT_LIBRARY_DIR}/libnvinfer.so
    ${TENSORRT_LIBRARY_DIR}/libnvonnxparser.so
)

# ---------- CuDNN (prebuilt in 3rdparty) ----------
set(CUDNN_ROOT         ${CMAKE_SOURCE_DIR}/3rdparty/cudnn89)
set(CUDNN_INCLUDE_DIR  ${CUDNN_ROOT}/include)
set(CUDNN_LIBRARY_DIR  ${CUDNN_ROOT}/lib64)

# Core libs: add nvinfer (you missed this), plugins, parsers
set(CUDNN_LIBRARIES
    ${CUDNN_LIBRARY_DIR}/libcudnn_adv_infer.so
    ${CUDNN_LIBRARY_DIR}/libcudnn_cnn_infer.so
    ${CUDNN_LIBRARY_DIR}/libcudnn_ops_infer.so
    ${CUDNN_LIBRARY_DIR}/libcudnn.so
)

# ---------- Project library ----------
add_library(integrate_trt SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/trt_wrapper/trt_infer_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/trt_wrapper/preprocess_cuda.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/object_detect/infer_2dod.cpp
)

target_include_directories(integrate_trt
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDNN_INCLUDE_DIR}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(integrate_trt
  PUBLIC
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARIES}
    ${CUDNN_LIBRARIES}
    Threads::Threads
    CUDA::cudart
)

# ---------- Test executable ----------
add_executable(test_2dod ${CMAKE_SOURCE_DIR}/test/test_2dod.cpp)
target_link_libraries(test_2dod PRIVATE integrate_trt)
