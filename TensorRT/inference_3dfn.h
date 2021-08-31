#ifndef TENSORRT_INFERENCE_3DFN_H
#define TENSORRT_INFERENCE_3DFN_H

#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "logger.h"
#include "util.h"

// Container for output of inference (3 outputs)
struct infer_output {
    float *keypoints;
    float *features;
    float *attention;
};

infer_output nms(float *xyz, float *attention); // non-max suppression

class Feat3dNetInference {
public:
    Feat3dNetInference(const std::string& engineFilename);
    bool infer(const std::string& input_filename, int32_t width, int32_t height, const std::string& output_filename);

private:
    std::string mEngineFilename;        // Filename of the serialized engine.

    nvinfer1::Dims mInputDims;          // The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims;         // The dimensions of the output to the network.

    util::UniquePtr<nvinfer1::ICudaEngine> mEngine; 
    // The TensorRT engine used to run the network
};

//!
//! \class SampleSegmentation
//!
//! \brief Implements semantic segmentation using FCN-ResNet101 ONNX model.
//!
class SampleSegmentation
{

public:
    SampleSegmentation(const std::string& engineFilename);
    bool infer(const std::string& input_filename, int32_t width, int32_t height, const std::string& output_filename);

private:
    std::string mEngineFilename;                    // Filename of the serialized engine.

    nvinfer1::Dims mInputDims;                      // The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims;                     // The dimensions of the output to the network.

    util::UniquePtr<nvinfer1::ICudaEngine> mEngine; // The TensorRT engine used to run the network
};




#endif // TENSORRT_INFERENCE_3DFN_H