/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

* Edited by Tianyi for inference with 3DFeatNet
*/

#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "util.h"

#include "inference_3dfn.h"

constexpr long long operator"" _MiB(long long unsigned val)
{
    return val * (1 << 20);
}

using sample::gLogError;
using sample::gLogInfo;
using sample::gLogVerbose;

/*
~ Command Line Arguments
~ --input [file1 file2 ... ] : list of files to run inference on.
~ --output [file] : optional arg detailing where to write output to.
~ --model [file] : optional arg detailing where to find the TensorRT engine.
*/

std::vector<std::string> INPUT_FILES;
std::string OUTPUT_DIR = "./TensorRT/infer_out";
std::string MODEL_PATH = "./TensorRT/model_infer.lib";
const int DIMS = 6;
int MAX_KEYPOINTS = 1024;

int main(int argc, char** argv)
{   
    // Extract input and output directories
    for (int i=1; i<argc; i++){
        if ( argv[i] == std::string("--model") && ++i<argc){
            MODEL_PATH = argv[i];
        } else if (argv[i] == std::string("--output") && ++i<argc){
            OUTPUT_DIR = argv[i];
        } else if ( argv[i] == std::string("--input") && ++i<argc){
            // Not out of bounds and does not start with '-'.
            // Relies on short-circuit eval!
            while(i < argc && argv[i+1][0]!='-'){
                INPUT_FILES.push_back( argv[i++] );
            }
        } else {
            gLogError << "Invalid flag passed " << argv[i] << "." << std::endl;
            gLogError << "Flags for inference\n"
            << "--input [file1 file2 ... ] : list of files to run inference on.\n"
            << "--output [file] : optional arg detailing where to write output to.\n"
            << "--model [file] : optional arg detailing where to find the TensorRT engine.\n"
            << std::endl;

            exit(1);
        }

    }

    // Check if vectors are not empty
    if (INPUT_FILES.size() == 0) {
        gLogError << "Input directories not defined. Exiting." << std::endl;
        exit(1);
    }

    gLogInfo << "Input Files: (" << INPUT_FILES.size() << "):" << std::endl;
    for (auto a : INPUT_FILES) gLogInfo << "    " << a << std::endl;
    gLogInfo << "Output Directory: " << OUTPUT_DIR << std::endl;

    gLogInfo << "Constructing Feat3dNet Inference model at path "
        << MODEL_PATH << std::endl;
    Feat3dNet model(MODEL_PATH);

    // Perform inference for each of the files
    for( auto input_file : INPUT_FILES ) {
        // Load point cloud as array.
        
        std::vector<float> pc;
        const int BATCH_SIZE = 1;
        const int descriptorDim = 32;   // num filters for attention
        const int POINT_DIM = 3;
        gLogInfo << "#### Processing bin file \'" << input_file << "\'..." << std::endl;
        const int NUM_POINTS = ReadVariableFromBin(pc, input_file, DIMS);
        gLogInfo << "#### Read float from bin file with length " << pc.size() 
            << " and num elements: " << pc.size()/DIMS << std::endl;

        // ? Randomise points (if necessary)
        // ? Downsample points (if necessary)
        
        // ! Only if RAM allows for it. If not, have to process in batches.
        // Compute attention in batches due to limited memory
        // Run inference here
        
        std::unique_ptr<float> pointcloudIn ( new float[BATCH_SIZE*NUM_POINTS*DIMS] );
        std::unique_ptr<float> keypointsIn ( new float[BATCH_SIZE*NUM_POINTS*POINT_DIM] );
        std::unique_ptr<float> featuresOut (new float[BATCH_SIZE*NUM_POINTS*descriptorDim]);
        std::unique_ptr<float> attentionOut (new float[BATCH_SIZE*NUM_POINTS]);

        // copy into model
        for(int i=0; i<NUM_POINTS; i++){
            pointcloudIn.get()[i*DIMS + 0] = pc[i*DIMS + 0];
            pointcloudIn.get()[i*DIMS + 1] = pc[i*DIMS + 1];
            pointcloudIn.get()[i*DIMS + 2] = pc[i*DIMS + 2];
            pointcloudIn.get()[i*DIMS + 3] = pc[i*DIMS + 3];
            pointcloudIn.get()[i*DIMS + 4] = pc[i*DIMS + 4];
            pointcloudIn.get()[i*DIMS + 5] = pc[i*DIMS + 5];

            keypointsIn.get()[i*POINT_DIM + 0] = pc[i*DIMS + 0];
            keypointsIn.get()[i*POINT_DIM + 1] = pc[i*DIMS + 1];
            keypointsIn.get()[i*POINT_DIM + 2] = pc[i*DIMS + 2];
        }

        gLogInfo << "#### Finished writing to input buffers." << std::endl;

        bool status = model.infer(pointcloudIn, keypointsIn, NUM_POINTS, DIMS, 
                                    featuresOut, attentionOut);

        if (!status) {
            gLogError << "Error in calling the forward pass!" << std::endl;
            exit(1);
        }

        gLogInfo << "Successfully ran inference for file " << input_file << std::endl;

        // nms to select keypoints based on attention

        // Compute features
        // Run inference here again

        // Save the output
    }

    // gLogInfo << "Running TensorRT inference for 3DFeatNet" << std::endl;
    return 0;
}
