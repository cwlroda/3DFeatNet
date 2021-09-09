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
#include <chrono>

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
const int BATCH_SIZE = 1;
const int descriptorDim = 32;   // num filters for attention
const int POINT_DIM = 3;
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
    // for (auto a : INPUT_FILES) gLogInfo << "    " << a << std::endl;
    gLogInfo << "Output Directory: " << OUTPUT_DIR << std::endl;

    gLogInfo << "Constructing Feat3dNet Inference model at path "
        << MODEL_PATH << std::endl;
    Feat3dNet model(MODEL_PATH);

    int fileIndex = 0;
    std::chrono::microseconds avgExecutionTime(0);
    int INFER_COUNT = 1500;
    INFER_COUNT = INPUT_FILES.size() < INFER_COUNT ? INPUT_FILES.size() : INFER_COUNT;

    // Perform inference for each of the files
    for( auto input_file : INPUT_FILES ) {
        gLogInfo << "#### Processing bin file [ " << fileIndex << " / " 
            << INPUT_FILES.size() << " ] \'" << input_file << "\'..." << std::endl;

        // Find size of bin file
        std::ifstream binFile(input_file, std::ios::binary);
        binFile.seekg(0, std::ifstream::end);
        auto FLEN = binFile.tellg() / sizeof(float);
        binFile.seekg(0, std::ifstream::beg);

        const int NUM_POINTS = FLEN / DIMS;

        gLogInfo << "#### Found float from bin file with length " << FLEN 
            << " and num elements: " << NUM_POINTS << std::endl;

        std::unique_ptr<float> pointcloudIn ( new float[NUM_POINTS*DIMS] );
        std::unique_ptr<float> keypointsIn ( new float[NUM_POINTS*POINT_DIM] );
        std::unique_ptr<float> featuresOut (new float[NUM_POINTS*descriptorDim]);
        std::unique_ptr<float> attentionOut (new float[NUM_POINTS]);

        float f;    // acc for current coordinate
        int pcIndex=0, kpIndex=0;
        while( binFile.read(reinterpret_cast<char*>(&f), sizeof(float)) ){
            pointcloudIn.get()[pcIndex] = f;
            // Only add to keypoints for the first 3 elements
            if (pcIndex % 6 < 3) keypointsIn.get()[kpIndex++] = f;
            
            pcIndex += 1;
        }

        // ? Randomise points (if necessary)
        // ? Downsample points (if necessary)
        //? Compute attention in batches due to limited memory
        //? Only if RAM allows for it. If not, have to process in batches.

        auto start = std::chrono::high_resolution_clock::now();
        bool status = model.infer(pointcloudIn, keypointsIn, NUM_POINTS, DIMS, 
                                    featuresOut, attentionOut);
        auto stop = std::chrono::high_resolution_clock::now();

        /*
            Pseudo-code for running separate inference:
            model.detectInfer();

            NMS

            model.describeInfer();
        */

        // free memory
        pointcloudIn.release();
        keypointsIn.release();
        featuresOut.release();
        attentionOut.release();

        if (!status) {
            gLogError << "Error in calling the forward pass!" << std::endl;
            exit(1);
        } else {
            auto infer_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
            gLogInfo << "#### Forward pass took " << infer_duration.count() << "us." << std::endl;
            avgExecutionTime += infer_duration;
        }

        gLogInfo << "Successfully ran inference for file " << input_file << std::endl << std::endl;
    
        if (fileIndex++ > INFER_COUNT){
            break;
        }
    }

    int avgExecution = avgExecutionTime.count()/fileIndex;
    gLogInfo << "On a total of " << fileIndex << " files, 3DFeatNet accelerated "
        << "by TensorRT took an average of " << avgExecution 
        << "us to run." << std::endl;

    return 0;
}
