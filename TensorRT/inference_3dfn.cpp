#include "inference_3dfn.h"

using sample::gLogError;
using sample::gLogInfo; // logger shenanigans
using sample::gLogVerbose;

Feat3dNet::Feat3dNet(const std::string& engineFilename)
    : mEngineFilename(engineFilename), mEngine(nullptr){

    // Get plugin from registry
    QueryBallPointPluginCreator qbpCreator;
    GroupPointPluginCreator gpCreator;
    SignOpPluginCreator sgnCreator;

    gLogInfo << "Adding custom plugins... "<< std::endl;
    initLibNvInferPlugins( getLogger(), qbpCreator.getPluginNamespace() );
    gLogInfo << "#### QueryBallPoint Namespace: " << qbpCreator.getPluginNamespace() << std::endl;
    initLibNvInferPlugins( getLogger(), gpCreator.getPluginNamespace() );
    gLogInfo << "#### GroupPoint Namespace: " << gpCreator.getPluginNamespace() << std::endl;
    initLibNvInferPlugins( getLogger(), sgnCreator.getPluginNamespace() );
    gLogInfo << "#### Sign_Op Namespace: " << sgnCreator.getPluginNamespace() << std::endl;

    // De-serialize engine from file
    std::ifstream engineFile(engineFilename, std::ios::binary);
    if (engineFile.fail()) {
        gLogError << "ERROR: Unable to deserialize engine from file." << std::endl;
        return;
    }

    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    util::UniquePtr<nvinfer1::IRuntime> runtime{
        nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())
    };
    mEngine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    assert(mEngine.get() != nullptr);
}

/* Runs the TensorRT inference for Feat3dNet.
    Allocate input and output memory, and executes the engine.
    Args:
    - in_points: 
*/
bool Feat3dNet::infer(std::unique_ptr<float> &aPointcloud,
                std::unique_ptr<float> &aKeypoints, 
                int32_t num_points, int32_t dims,
                std::unique_ptr<float> &features_buffer,
                std::unique_ptr<float> &attention_buffer
        ) {
    auto context = 
        util::UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context) {
        gLogError << "Error creating execution context from engine." << std::endl;
        return false;
    }
    /*
    3DFeatNet has one input and three output tensors.
    Inputs:
        pointcloud
    Outputs:
        keypoints
        features
        attention

    Input data is passed in as {}.
    */

    gLogInfo << "#### Input dimensions for point cloud: [" << 1 << ", " << num_points 
                << ", " << dims << "]\n";

    // auto profile = context->getOptimizationProfile();
    auto nbProfiles = mEngine->getNbOptimizationProfiles();
    gLogInfo << "#### Execution context has " << nbProfiles << " optimization profiles." << std::endl;

    auto expectedDims = context->getBindingDimensions(1);
    for (int i=0; i<expectedDims.nbDims; i++){
        gLogInfo << "Pointcloud dimension [" << i << "]: " << expectedDims.d[i] << std::endl;
    }


    //~ get bindings for input and outputs
    // get input bindings
    auto pointcloud_idx = mEngine->getBindingIndex("in_pointcloud");
    if (pointcloud_idx == -1) {
        gLogError << "Unable to find input with name 'in_pointcloud'." << std::endl;
        return false;
    }
    assert(mEngine->getBindingDataType(pointcloud_idx) == nvinfer1::DataType::kFLOAT);
    auto pointcloud_dims = nvinfer1::Dims3{1, num_points, dims};
    context->setBindingDimensions(pointcloud_idx, pointcloud_dims);
    auto pointcloud_size = util::getMemorySize(pointcloud_dims, sizeof(float));

    // get bindings for Keypoints
    auto keypoints_idx = mEngine->getBindingIndex("in_keypoints");
    if (keypoints_idx == -1) {
        gLogError << "Unable to find output with name 'in_keypoints'." << std::endl;
        return false;
    }
    assert(mEngine->getBindingDataType(keypoints_idx) == nvinfer1::DataType::kFLOAT);
    auto keypoint_dims = nvinfer1::Dims3{1, num_points, 3};
    context->setBindingDimensions(keypoints_idx, keypoint_dims);
    auto keypoints_size = util::getMemorySize(keypoint_dims, sizeof(float));

    // get bindings for Features
    auto features_idx = mEngine->getBindingIndex("out_features");
    if (features_idx == -1) {
        gLogError << "Unable to find output with name 'out_features'." << std::endl;
        return false;
    }
    assert(mEngine->getBindingDataType(features_idx) == nvinfer1::DataType::kFLOAT);
    auto features_dims = context->getBindingDimensions(features_idx);
    auto features_size = util::getMemorySize(features_dims, sizeof(float));

    // get bindings for Attention
    auto attention_idx = mEngine->getBindingIndex("out_attention");
    if (attention_idx == -1) {
        gLogError << "Unable to find output with name 'out_attention'." << std::endl;
        return false;
    }
    assert(mEngine->getBindingDataType(attention_idx) == nvinfer1::DataType::kFLOAT);
    auto attention_dims = context->getBindingDimensions(attention_idx);
    auto attention_size = util::getMemorySize(attention_dims, sizeof(float));
    //~ end get bindings for input and outputs

    gLogInfo << "#### Attempting to allocate CUDA memory..." << std::endl;

    //~ Allocate CUDA memory for input and output bindings
    void* pointcloud_mem{nullptr};
    if (cudaMalloc(&pointcloud_mem, pointcloud_size) != cudaSuccess) {
        gLogError << "ERROR: pointcloud cuda memory allocation failed, size = " 
            << pointcloud_size << " bytes" << std::endl;
        return false;
    }
    void* keypoints_mem{nullptr};
    if (cudaMalloc(&keypoints_mem, keypoints_size) != cudaSuccess){
        gLogError << "ERROR: keypoints cuda memory allocation failed, size = " 
            << keypoints_size << " bytes" << std::endl;
        return false;
    }
    void* features_mem{nullptr};
    if (cudaMalloc(&features_mem, features_size) != cudaSuccess){
        gLogError << "ERROR: features cuda memory allocation failed, size = " 
            << features_size << " bytes" << std::endl;
        return false;
    }
    void* attention_mem{nullptr};
    if (cudaMalloc(&attention_mem, attention_size) != cudaSuccess){
        gLogError << "ERROR: attention cuda memory allocation failed, size = " 
            << attention_size << " bytes" << std::endl;
        return false;
    }
    //~ End allocate CUDA memory for input and output bindings

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return false;
    }

    // Copy point cloud data to input binding memory
    if (cudaMemcpyAsync(pointcloud_mem, aPointcloud.get()
    , pointcloud_size, 
            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        gLogError << "ERROR: CUDA memory copy of pointcloud failed, size = " 
            << pointcloud_size << " bytes" << std::endl;
        return false;
    }

    // Copy keypoint data to input binding memory
    if (cudaMemcpyAsync(keypoints_mem, aKeypoints.get(), keypoints_size, 
            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        gLogError << "ERROR: CUDA memory copy of keypoints failed, size = " 
            << keypoints_size << " bytes" << std::endl;
        return false;
    }

    cudaStreamSynchronize(stream);

    // Run TensorRT inference
    void* bindings[] = {pointcloud_mem, keypoints_mem, features_mem, attention_mem};
    bool status = context->enqueueV2(bindings, stream, nullptr);
    if (!status) {
        gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    }

    //~ Copy predictions from output binding memory.
    // Copy features
    if (cudaMemcpyAsync(features_buffer.get(), features_mem, features_size, 
        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        gLogError << "ERROR: CUDA memory copy of features failed, size = "
            << features_size << " bytes" << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);

    // Copy attention
    if (cudaMemcpyAsync(attention_buffer.get(), attention_mem, attention_size, 
        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        gLogError << "ERROR: CUDA memory copy of attention failed, size = "
            << attention_size << " bytes" << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);

    // Free CUDA resources
    cudaFree(pointcloud_mem);
    cudaFree(keypoints_mem);
    cudaFree(features_mem);
    cudaFree(attention_mem);
    return true;
}

// Reads a float32 from the input file.
int ReadVariableFromBin(std::vector<float> &arr,
                        std::string filename, const int dims){
    float f;
    std::ifstream f_in(filename, std::ios::binary);

    int dims_cnt=0;
    int rows = 0;
    while( f_in.read(reinterpret_cast<char*>(&f), sizeof(float)) ){
        // arr.get()[ rows*dims + dims_cnt++] = f;
        arr.push_back(f);

        // dims_cnt += 1;
        // if ( !(dims_cnt < dims) ){
        //     dims_cnt = 0;
        //     rows += 1;
        // }
    }

    return arr.size() / dims;
}
