#ifndef TENSORRT_INFERENCE_3DFN_H
#define TENSORRT_INFERENCE_3DFN_H

#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <chrono>

// #include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "logger.h"
#include "util.h"

#include "noisy_assert.h"
#include "ops/grouping/grouping_plugin.h"
#include "ops/sign/sign_op_plugin.h"

// Reads point cloud from binary file into 1D vector.
int ReadVariableFromBin(std::vector<float> &arr, 
            std::string filename, const int dims);

class Feat3dNet {
public:
    Feat3dNet(const std::string& engineFilename);
    bool infer(std::unique_ptr<float> &aPointcloud,
                std::unique_ptr<float> &aKeypoints, 
                int32_t num_points, int32_t dims,
                std::unique_ptr<float> &features_buffer,
                std::unique_ptr<float> &attention_buffer
                );

private:
    std::string mEngineFilename;        // Filename of the serialized engine.
    nvinfer1::Dims mInputDims;          // The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims;         // The dimensions of the output to the network.
    util::UniquePtr<nvinfer1::ICudaEngine> mEngine;
    util::UniquePtr<nvinfer1::IExecutionContext> mContext;
    // The TensorRT engine used to run the network
};

// Defines a 3DFeatNet architecture with separately-callable detect and describe networks.
class Feat3dNet_segemented{
public:
    Feat3dNet_segemented(const std::string& detectFilename, const std::string& describeFilename);
    
    // detector inference, same inputs and outputs as defined in the ONNX model.
    bool detectInfer(    
        std::unique_ptr<float> &aPointcloud,
        int32_t num_points, int32_t dims,
        std::unique_ptr<float> &keypoints_buffer,
        std::unique_ptr<float> &attention_buffer,
        std::unique_ptr<float> &orientation_buffer
    );

    // descriptor inference, same inputs and outputs as defined in the ONNX model.
    bool describeInfer(
        std::unique_ptr<float> &aPointcloud,
        std::unique_ptr<float> &aKeypoints,
        std::unique_ptr<float> &aOrientation,
        int32_t num_points, int32_t dims,
        std::unique_ptr<float> &keypoints_buffer,
        std::unique_ptr<float> &features_buffer
    );

private:
    std::string detectEngineFilename, describeEngineFilename;   // Filename of the serialized engine.
    // nvinfer1::Dims mInputDims;          // The dimensions of the input to the network.
    // nvinfer1::Dims mOutputDims;         // The dimensions of the output to the network.
    util::UniquePtr<nvinfer1::ICudaEngine> detectEngine, describeEngine;
    util::UniquePtr<nvinfer1::IExecutionContext> detectContext, describeContext;
    // The TensorRT engine used to run the network

#endif // TENSORRT_INFERENCE_3DFN_H