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

#include "noisy_assert.h"

// #include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

extern int MAX_KEYPOINTS;   // for use elsewhere

typedef Eigen::Tensor<float, 3> pointcloud_t;
// typedef std::vector<std::vector<std::vector<float>>> pointcloud_t;

// Container for output of inference (3 outputs)
struct infer_output {
    float *keypoints, *features, *attention;
    int32_t num_points, dimension=3;
    // float *keypoints, *features, *attention;
    // int keypoints_size, features_size, attention_size;
};

// non-max suppression
infer_output nms(pointcloud_t &keypoints, pointcloud_t &attention); 

// Reads point cloud from binary file into 1D vector.
void ReadVariableFromBin(std::vector<float> &vect, 
            std::string filename, const int dims=6);

class Feat3dNetInference {
public:
    Feat3dNetInference(const std::string& engineFilename);
    bool infer(Eigen::TensorMap<pointcloud_t> &in_points, int32_t num_points,
            int32_t dims, infer_output out_points);

private:
    std::string mEngineFilename;        // Filename of the serialized engine.
    nvinfer1::Dims mInputDims;          // The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims;         // The dimensions of the output to the network.
    util::UniquePtr<nvinfer1::ICudaEngine> mEngine; 
    // The TensorRT engine used to run the network
};

/*
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
*/

#endif // TENSORRT_INFERENCE_3DFN_H