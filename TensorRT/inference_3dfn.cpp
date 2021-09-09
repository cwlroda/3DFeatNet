#include "inference_3dfn.h"

using sample::gLogError;
using sample::gLogInfo; // logger shenanigans
using sample::gLogVerbose;

Feat3dNet::Feat3dNet(const std::string& engineFilename)
    : mEngineFilename(engineFilename), mEngine(nullptr), mContext(nullptr){

    // Get plugin from registry
    QueryBallPointPluginCreator qbpCreator;
    GroupPointPluginCreator gpCreator;
    SignOpPluginCreator sgnCreator;

    gLogVerbose << "Adding custom plugins... "<< std::endl;
    initLibNvInferPlugins( getLogger(), qbpCreator.getPluginNamespace() );
    gLogVerbose << "#### QueryBallPoint Namespace: " << qbpCreator.getPluginNamespace() << std::endl;
    initLibNvInferPlugins( getLogger(), gpCreator.getPluginNamespace() );
    gLogVerbose << "#### GroupPoint Namespace: " << gpCreator.getPluginNamespace() << std::endl;
    initLibNvInferPlugins( getLogger(), sgnCreator.getPluginNamespace() );
    gLogVerbose << "#### Sign_Op Namespace: " << sgnCreator.getPluginNamespace() << std::endl;

    // De-serialize engine from file
    std::ifstream engineFile(engineFilename, std::ios::binary);
    if (engineFile.fail()) {
        gLogError << "ERROR: Unable to deserialize engine from file." << std::endl;
        exit(1);
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

    mContext.reset( mEngine->createExecutionContext() );
    assert(mContext.get() != nullptr);
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
    // auto context = 
    //     util::UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    // if (!context) {
    //     gLogError << "Error creating execution context from engine." << std::endl;
    //     return false;
    // }
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

    gLogVerbose << "#### Input dimensions for point cloud: [" << 1 << ", " << num_points 
                << ", " << dims << "]\n";

    // auto profile = context->getOptimizationProfile();
    // auto nbProfiles = mEngine->getNbOptimizationProfiles();
    // gLogVerbose << "#### Execution context has " << nbProfiles << " optimization profiles." << std::endl;

    // auto expectedDims = context->getBindingDimensions(1);
    // for (int i=0; i<expectedDims.nbDims; i++){
    //     gLogVerbose << "Pointcloud dimension [" << i << "]: " << expectedDims.d[i] << std::endl;
    // }


    //~ get bindings for input and outputs
    // get input bindings
    auto pointcloud_idx = mEngine->getBindingIndex("in_pointcloud");
    if (pointcloud_idx == -1) {
        gLogError << "Unable to find input with name 'in_pointcloud'." << std::endl;
        return false;
    }
    assert(mEngine->getBindingDataType(pointcloud_idx) == nvinfer1::DataType::kFLOAT);
    auto pointcloud_dims = nvinfer1::Dims3{1, num_points, dims};
    mContext->setBindingDimensions(pointcloud_idx, pointcloud_dims);
    auto pointcloud_size = util::getMemorySize(pointcloud_dims, sizeof(float));

    // get bindings for Keypoints
    auto keypoints_idx = mEngine->getBindingIndex("in_keypoints");
    if (keypoints_idx == -1) {
        gLogError << "Unable to find output with name 'in_keypoints'." << std::endl;
        return false;
    }
    assert(mEngine->getBindingDataType(keypoints_idx) == nvinfer1::DataType::kFLOAT);
    auto keypoint_dims = nvinfer1::Dims3{1, num_points, 3};
    mContext->setBindingDimensions(keypoints_idx, keypoint_dims);
    auto keypoints_size = util::getMemorySize(keypoint_dims, sizeof(float));
    
    // get bindings for out_keypoints
    auto kp_out_idx = mEngine->getBindingIndex("out_keypoints");
    if (kp_out_idx == -1) {
        gLogError << "Unable to find output with name 'out_keypoints'." << std::endl;
        return false;
    }
    assert(mEngine->getBindingDataType(kp_out_idx) == nvinfer1::DataType::kFLOAT);
    auto kp_out_dims = mContext->getBindingDimensions(kp_out_idx);
    auto kp_out_size = util::getMemorySize(kp_out_dims, sizeof(float));

    // get bindings for Features
    auto features_idx = mEngine->getBindingIndex("out_features");
    if (features_idx == -1) {
        gLogError << "Unable to find output with name 'out_features'." << std::endl;
        return false;
    }
    assert(mEngine->getBindingDataType(features_idx) == nvinfer1::DataType::kFLOAT);
    auto features_dims = mContext->getBindingDimensions(features_idx);
    auto features_size = util::getMemorySize(features_dims, sizeof(float));

    // get bindings for Attention
    auto attention_idx = mEngine->getBindingIndex("out_attention");
    if (attention_idx == -1) {
        gLogError << "Unable to find output with name 'out_attention'." << std::endl;
        return false;
    }
    assert(mEngine->getBindingDataType(attention_idx) == nvinfer1::DataType::kFLOAT);
    auto attention_dims = mContext->getBindingDimensions(attention_idx);
    auto attention_size = util::getMemorySize(attention_dims, sizeof(float));
    //~ end get bindings for input and outputs

    gLogVerbose << "#### Attempting to allocate CUDA memory..." << std::endl;

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
    void* kp_out_mem{nullptr};
    if (cudaMalloc(&kp_out_mem, kp_out_size) != cudaSuccess){
        gLogError << "ERROR: out_keypoints cuda memory allocation failed, size = " 
            << kp_out_size << " bytes" << std::endl;
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

    auto start_mcpy1 = std::chrono::high_resolution_clock::now();
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
    auto stop_mcpy1 = std::chrono::high_resolution_clock::now();
    auto mcpy1_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_mcpy1-start_mcpy1);
    gLogInfo << "%%%% Copying args took " << mcpy1_duration.count() << "ms." << std::endl;        

    cudaStreamSynchronize(stream);

    auto start = std::chrono::high_resolution_clock::now();
    //~ Run TensorRT inference
    //! For testing, found that all inputs/output tensors must be represented
    //! here with the proper dimensions!
    //? Perhaps all bindings are not needed, and some CUDA memory can be saved??
    void* bindings[] = {keypoints_mem, pointcloud_mem, kp_out_mem, features_mem, attention_mem};
    bool status = mContext->enqueueV2(bindings, stream, nullptr);
    if (!status) {
        gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    } else {
        gLogVerbose << "TensorRT Inference successful." << std::endl;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
    gLogInfo << "%%%% Computation took " << infer_duration.count() << "ms." << std::endl;        

    //~ Copy predictions from output binding memory.
    auto start_mcpy2 = std::chrono::high_resolution_clock::now();
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

    auto stop_mcpy2 = std::chrono::high_resolution_clock::now();
    auto mcpy2_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_mcpy2-start_mcpy2);
    gLogInfo << "%%%% Copying output took " << mcpy2_duration.count() << "ms." << std::endl; 

    // Free CUDA resources
    cudaFree(pointcloud_mem);
    cudaFree(keypoints_mem);
    cudaFree(kp_out_mem);
    cudaFree(features_mem);
    cudaFree(attention_mem);
    return true;
}


Feat3dNet_segemented::Feat3dNet_segemented(
        const std::string& detectFilename, const std::string& describeFilename) :
        detectEngineFilename(detectFilename), describeEngineFilename(describeFilename),
        detectEngine(nullptr), describeEngine(nullptr), 
        detectContext(nullptr), describeContext(nullptr) {

    // Get plugin from registry
    QueryBallPointPluginCreator qbpCreator;
    GroupPointPluginCreator gpCreator;
    SignOpPluginCreator sgnCreator;

    gLogVerbose << "Adding custom plugins... "<< std::endl;
    initLibNvInferPlugins( getLogger(), qbpCreator.getPluginNamespace() );
    gLogVerbose << "#### QueryBallPoint Namespace: " << qbpCreator.getPluginNamespace() << std::endl;
    initLibNvInferPlugins( getLogger(), gpCreator.getPluginNamespace() );
    gLogVerbose << "#### GroupPoint Namespace: " << gpCreator.getPluginNamespace() << std::endl;
    initLibNvInferPlugins( getLogger(), sgnCreator.getPluginNamespace() );
    gLogVerbose << "#### Sign_Op Namespace: " << sgnCreator.getPluginNamespace() << std::endl;

    // De-serialize engines from file
    // Detector Engine
    std::ifstream detectEngineFile(detectEngineFilename, std::ios::binary);
    if (detectEngineFile.fail()) {
        gLogError << "ERROR: Unable to deserialize detection engine from file." << std::endl;
        exit(1);
    }

    detectEngineFile.seekg(0, std::ifstream::end);
    auto fsize = detectEngineFile.tellg();
    detectEngineFile.seekg(0, std::ifstream::beg);

    std::vector<char> detectEngineData(fsize);
    detectEngineFile.read(detectEngineData.data(), fsize);

    // Descriptor Engine
    std::ifstream describeEngineFile(describeEngineFilename, std::ios::binary);
    if (describeEngineFile.fail()) {
        gLogError << "ERROR: Unable to deserialize descriptor engine from file." << std::endl;
        exit(1);
    }

    describeEngineFile.seekg(0, std::ifstream::end);
    auto fsize = describeEngineFile.tellg();
    describeEngineFile.seekg(0, std::ifstream::beg);

    std::vector<char> describeEngineData(fsize);
    describeEngineFile.read(describeEngineData.data(), fsize);

    util::UniquePtr<nvinfer1::IRuntime> runtime{
        nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())
    };
    
    // Create pointers to engine, and execution contexts
    detectEngine.reset(runtime->deserializeCudaEngine(detectEngineData.data(), fsize, nullptr));
    assert(detectEngine.get() != nullptr);

    describeEngine.reset(runtime->deserializeCudaEngine(describeEngineData.data(), fsize, nullptr));
    assert(describeEngine.get() != nullptr);

    detectContext.reset( detectEngine->createExecutionContext() );
    assert(detectContext.get() != nullptr);

    describeContext.reset( describeEngine->createExecutionContext() );
    assert(describeContext.get() != nullptr);
}

bool Feat3dNet_segemented::detectInfer(
    std::unique_ptr<float> &aPointcloud,
    int32_t num_points, int32_t dims,
    std::unique_ptr<float> &keypoints_buffer,
    std::unique_ptr<float> &attention_buffer,
    std::unique_ptr<float> &orientation_buffer
){
    //~ get bindings for input and outputs
    // get input bindings
    auto pointcloud_idx = detectEngine->getBindingIndex("in_pointcloud");
    if (pointcloud_idx == -1) {
        gLogError << "Unable to find input with name 'in_pointcloud'." << std::endl;
        return false;
    }
    assert(detectEngine->getBindingDataType(pointcloud_idx) == nvinfer1::DataType::kFLOAT);
    auto pointcloud_dims = nvinfer1::Dims3{1, num_points, dims};
    detectContext->setBindingDimensions(pointcloud_idx, pointcloud_dims);
    auto pointcloud_size = util::getMemorySize(pointcloud_dims, sizeof(float));

    // get bindings for Features
    auto keypoints_idx = detectEngine->getBindingIndex("out_keypoints");
    if (keypoints_idx == -1) {
        gLogError << "Unable to find output with name 'out_keypoints'." << std::endl;
        return false;
    }
    assert(detectEngine->getBindingDataType(keypoints_idx) == nvinfer1::DataType::kFLOAT);
    auto keypoints_dims = detectContext->getBindingDimensions(keypoints_idx);
    auto keypoints_size = util::getMemorySize(keypoints_dims, sizeof(float));

    // get bindings for Attention
    auto attention_idx = detectEngine->getBindingIndex("out_attention");
    if (attention_idx == -1) {
        gLogError << "Unable to find output with name 'out_attention'." << std::endl;
        return false;
    }
    assert(detectEngine->getBindingDataType(attention_idx) == nvinfer1::DataType::kFLOAT);
    auto attention_dims = detectContext->getBindingDimensions(attention_idx);
    auto attention_size = util::getMemorySize(attention_dims, sizeof(float));

    // get bindings for Orientation
    auto orientation_idx = detectEngine->getBindingIndex("out_orientation");
    if (orientation_idx == -1) {
        gLogError << "Unable to find output with name 'out_orientation'." << std::endl;
        return false;
    }
    assert(detectEngine->getBindingDataType(orientation_idx) == nvinfer1::DataType::kFLOAT);
    auto orientation_dims = detectContext->getBindingDimensions(orientation_idx);
    auto orientation_size = util::getMemorySize(orientation_dims, sizeof(float));
    //~ end get bindings for input and outputs

    gLogVerbose << "#### Attempting to allocate CUDA memory..." << std::endl;

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
    void* attention_mem{nullptr};
    if (cudaMalloc(&attention_mem, attention_size) != cudaSuccess){
        gLogError << "ERROR: attention cuda memory allocation failed, size = " 
            << attention_size << " bytes" << std::endl;
        return false;
    }
    void* orientation_mem{nullptr};
    if (cudaMalloc(&orientation_mem, orientation_size) != cudaSuccess){
        gLogError << "ERROR: orientation cuda memory allocation failed, size = " 
            << orientation_size << " bytes" << std::endl;
        return false;
    }
    //~ End allocate CUDA memory for input and output bindings

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return false;
    }

    auto start_mcpy1 = std::chrono::high_resolution_clock::now();
    // Copy point cloud data to input binding memory
    if (cudaMemcpyAsync(pointcloud_mem, aPointcloud.get()
    , pointcloud_size, 
            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        gLogError << "ERROR: CUDA memory copy of pointcloud failed, size = " 
            << pointcloud_size << " bytes" << std::endl;
        return false;
    }
    auto stop_mcpy1 = std::chrono::high_resolution_clock::now();
    auto mcpy1_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_mcpy1-start_mcpy1);
    gLogInfo << "%%%% Copying args took " << mcpy1_duration.count() << "ms." << std::endl;        

    cudaStreamSynchronize(stream);

    auto start = std::chrono::high_resolution_clock::now();
    //~ Run TensorRT inference
    //! For testing, found that all inputs/output tensors must be represented
    //! here with the proper dimensions!
    //? Perhaps all bindings are not needed, and some CUDA memory can be saved??
    void* bindings[] = {pointcloud_mem, keypoints_mem, attention_mem, orientation_mem};
    bool status = detectContext->enqueueV2(bindings, stream, nullptr);
    if (!status) {
        gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    } else {
        gLogVerbose << "TensorRT Inference successful." << std::endl;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
    gLogInfo << "%%%% Computation took " << infer_duration.count() << "ms." << std::endl;        

    //~ Copy predictions from output binding memory.
    auto start_mcpy2 = std::chrono::high_resolution_clock::now();
    // Copy features
    if (cudaMemcpyAsync(keypoints_buffer.get(), keypoints_mem, keypoints_size, 
        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        gLogError << "ERROR: CUDA memory copy of keypoints failed, size = "
            << keypoints_size << " bytes" << std::endl;
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

    // Copy orientation
    if (cudaMemcpyAsync(orientation_buffer.get(), orientation_mem, orientation_size, 
        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        gLogError << "ERROR: CUDA memory copy of orientation failed, size = "
            << orientation_size << " bytes" << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);

    auto stop_mcpy2 = std::chrono::high_resolution_clock::now();
    auto mcpy2_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_mcpy2-start_mcpy2);
    gLogInfo << "%%%% Copying output took " << mcpy2_duration.count() << "ms." << std::endl; 

    // Free CUDA resources
    cudaFree(pointcloud_mem);
    cudaFree(keypoints_mem);
    cudaFree(attention_mem);
    cudaFree(orientation_mem);
    return true;
}

bool Feat3dNet_segemented::describeInfer(        
    std::unique_ptr<float> &aPointcloud,
    std::unique_ptr<float> &aKeypoints,
    std::unique_ptr<float> &aOrientation,
    int32_t n_pointcloud, int32_t n_keypoints, int32_t dims,
    std::unique_ptr<float> &keypoints_buffer,
    std::unique_ptr<float> &features_buffer
){
    //~ get bindings for input and outputs
    // get input bindings
    auto pointcloud_idx = describeEngine->getBindingIndex("in_pointcloud");
    if (pointcloud_idx == -1) {
        gLogError << "Unable to find input with name 'in_pointcloud'." << std::endl;
        return false;
    }
    assert(describeEngine->getBindingDataType(pointcloud_idx) == nvinfer1::DataType::kFLOAT);
    auto pointcloud_dims = nvinfer1::Dims3{1, n_pointcloud, dims};
    describeContext->setBindingDimensions(pointcloud_idx, pointcloud_dims);
    auto pointcloud_size = util::getMemorySize(pointcloud_dims, sizeof(float));

    // get bindings for Keypoints
    auto keypoints_idx = describeEngine->getBindingIndex("in_keypoints");
    if (keypoints_idx == -1) {
        gLogError << "Unable to find output with name 'in_keypoints'." << std::endl;
        return false;
    }
    assert(describeEngine->getBindingDataType(keypoints_idx) == nvinfer1::DataType::kFLOAT);
    auto keypoint_dims = nvinfer1::Dims3{1, n_keypoints, 3};
    describeContext->setBindingDimensions(keypoints_idx, keypoint_dims);
    auto keypoints_size = util::getMemorySize(keypoint_dims, sizeof(float));
    
    // get bindings for Orientation
    auto orientation_idx = describeEngine->getBindingIndex("in_orientation");
    if (orientation_idx == -1) {
        gLogError << "Unable to find output with name 'in_orientation'." << std::endl;
        return false;
    }
    assert(describeEngine->getBindingDataType(orientation_idx) == nvinfer1::DataType::kFLOAT);
    auto orientation_dims = nvinfer1::Dims3{1, n_pointcloud, 3};
    describeContext->setBindingDimensions(orientation_idx, orientation_dims);
    auto orientation_size = util::getMemorySize(orientation_dims, sizeof(float));

    // get bindings for out_keypoints
    auto kp_out_idx = describeEngine->getBindingIndex("out_keypoints");
    if (kp_out_idx == -1) {
        gLogError << "Unable to find output with name 'out_keypoints'." << std::endl;
        return false;
    }
    assert(describeEngine->getBindingDataType(kp_out_idx) == nvinfer1::DataType::kFLOAT);
    auto kp_out_dims = describeContext->getBindingDimensions(kp_out_idx);
    auto kp_out_size = util::getMemorySize(kp_out_dims, sizeof(float));

    // get bindings for Features
    auto features_idx = describeEngine->getBindingIndex("out_features");
    if (features_idx == -1) {
        gLogError << "Unable to find output with name 'out_features'." << std::endl;
        return false;
    }
    assert(describeEngine->getBindingDataType(features_idx) == nvinfer1::DataType::kFLOAT);
    auto features_dims = describeContext->getBindingDimensions(features_idx);
    auto features_size = util::getMemorySize(features_dims, sizeof(float));
    //~ end get bindings for input and outputs

    gLogVerbose << "#### Attempting to allocate CUDA memory..." << std::endl;

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
    void* orientation_mem{nullptr};
    if (cudaMalloc(&orientation_mem, orientation_size) != cudaSuccess){
        gLogError << "ERROR: orientaiton cuda memory allocation failed, size = " 
            << orientation_size << " bytes" << std::endl;
        return false;
    }
    void* kp_out_mem{nullptr};
    if (cudaMalloc(&kp_out_mem, kp_out_size) != cudaSuccess){
        gLogError << "ERROR: out_keypoints cuda memory allocation failed, size = " 
            << kp_out_size << " bytes" << std::endl;
        return false;
    }
    void* features_mem{nullptr};
    if (cudaMalloc(&features_mem, features_size) != cudaSuccess){
        gLogError << "ERROR: features cuda memory allocation failed, size = " 
            << features_size << " bytes" << std::endl;
        return false;
    }
    //~ End allocate CUDA memory for input and output bindings

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return false;
    }

    auto start_mcpy1 = std::chrono::high_resolution_clock::now();
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

    // Copy orientation data to input binding memory
    if (cudaMemcpyAsync(orientation_mem, aOrientation.get(), orientation_size, 
            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        gLogError << "ERROR: CUDA memory copy of orientation failed, size = " 
            << orientation_size << " bytes" << std::endl;
        return false;
    }
    auto stop_mcpy1 = std::chrono::high_resolution_clock::now();
    auto mcpy1_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_mcpy1-start_mcpy1);
    gLogInfo << "%%%% Copying args took " << mcpy1_duration.count() << "ms." << std::endl;        

    cudaStreamSynchronize(stream);

    auto start = std::chrono::high_resolution_clock::now();
    //~ Run TensorRT inference
    //! For testing, found that all inputs/output tensors must be represented
    //! here with the proper dimensions!
    //? Perhaps all bindings are not needed, and some CUDA memory can be saved??
    void* bindings[] = {keypoints_mem, pointcloud_mem, orientation_mem, kp_out_mem, features_mem};
    bool status = describeContext->enqueueV2(bindings, stream, nullptr);
    if (!status) {
        gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    } else {
        gLogVerbose << "TensorRT Inference successful." << std::endl;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
    gLogInfo << "%%%% Computation took " << infer_duration.count() << "ms." << std::endl;        

    //~ Copy predictions from output binding memory.
    auto start_mcpy2 = std::chrono::high_resolution_clock::now();
    // Copy features
    if (cudaMemcpyAsync(features_buffer.get(), features_mem, features_size, 
        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        gLogError << "ERROR: CUDA memory copy of features failed, size = "
            << features_size << " bytes" << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);

    // Copy attention
    if (cudaMemcpyAsync(keypoints_buffer.get(), kp_out_mem, kp_out_size, 
        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        gLogError << "ERROR: CUDA memory copy of keypoints failed, size = "
            << kp_out_size << " bytes" << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);

    auto stop_mcpy2 = std::chrono::high_resolution_clock::now();
    auto mcpy2_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_mcpy2-start_mcpy2);
    gLogInfo << "%%%% Copying output took " << mcpy2_duration.count() << "ms." << std::endl; 

    // Free CUDA resources
    cudaFree(pointcloud_mem);
    cudaFree(keypoints_mem);
    cudaFree(orientation_mem);
    cudaFree(kp_out_mem);
    cudaFree(features_mem);
    return true;
}


// Reads a float32 from the input file (unused utility).
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
