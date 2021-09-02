#include "inference_3dfn.h"

using sample::gLogError;
using sample::gLogInfo; // logger shenanigans

Feat3dNetInference::Feat3dNetInference(const std::string& engineFilename)
    : mEngineFilename(engineFilename), mEngine(nullptr) {
    
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

    util::UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};
    mEngine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    assert(mEngine.get() != nullptr);
}

// Runs the TensorRT inference for Feat3dNet.
// Allocate input and output memory, and executes the engine.
bool Feat3dNetInference::infer(Eigen::TensorMap<pointcloud_t> &in_points, int32_t num_points,
        int32_t dims, infer_output out_points) {
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

    //~ get bindings for input and outputs
    // get input bindings
    auto input_idx = mEngine->getBindingIndex("pointcloud");
    if (input_idx == -1) {
        gLogError << "Unable to find input with name 'pointcloud'." << std::endl;
        return false;
    }

    assert(mEngine->getBindingDataType(input_idx) == nvinfer1::DataType::kFLOAT);
    auto input_dims = nvinfer1::Dims3{1, num_points, dims};
    context->setBindingDimensions(input_idx, input_dims);
    auto input_size = util::getMemorySize(input_dims, sizeof(float));

    // get bindings for Keypoints
    auto keypoints_idx = mEngine->getBindingIndex("keypoints");
    if (keypoints_idx == -1) {
        gLogError << "Unable to find output with name 'keypoints'." << std::endl;
        return false;
    }
    assert(mEngine->getBindingDataType(keypoints_idx) == nvinfer1::DataType::kFLOAT);
    auto keypoints_dims = context->getBindingDimensions(keypoints_idx);
    auto keypoints_size = util::getMemorySize(keypoints_dims, sizeof(float));

    // get bindings for Features
    auto features_idx = mEngine->getBindingIndex("features");
    if (features_idx == -1) {
        gLogError << "Unable to find output with name 'features'." << std::endl;
        return false;
    }
    assert(mEngine->getBindingDataType(features_idx) == nvinfer1::DataType::kFLOAT);
    auto features_dims = context->getBindingDimensions(features_idx);
    auto features_size = util::getMemorySize(features_dims, sizeof(float));

    // get bindings for Attention
    auto attention_idx = mEngine->getBindingIndex("attention");
    if (attention_idx == -1) {
        gLogError << "Unable to find output with name 'attention'." << std::endl;
        return false;
    }
    assert(mEngine->getBindingDataType(attention_idx) == nvinfer1::DataType::kFLOAT);
    auto attention_dims = context->getBindingDimensions(attention_idx);
    auto attention_size = util::getMemorySize(attention_dims, sizeof(float));
    //~ end get bindings for input and outputs

    //~ Allocate CUDA memory for input and output bindings
    void* input_mem{nullptr};
    if (cudaMalloc(&input_mem, input_size) != cudaSuccess) {
        gLogError << "ERROR: input cuda memory allocation failed, size = " 
            << input_size << " bytes" << std::endl;
        return false;
    }
    void* keypoints_mem{nullptr};
    if (cudaMalloc(&keypoints_mem, keypoints_size) != cudaSuccess){
        gLogError << "ERROR: output cuda memory allocation failed, size = " 
            << keypoints_size << " bytes" << std::endl;
        return false;
    }
    void* features_mem{nullptr};
    if (cudaMalloc(&features_mem, features_size) != cudaSuccess){
        gLogError << "ERROR: output cuda memory allocation failed, size = " 
            << features_size << " bytes" << std::endl;
        return false;
    }
    void* attention_mem{nullptr};
    if (cudaMalloc(&attention_mem, attention_size) != cudaSuccess){
        gLogError << "ERROR: output cuda memory allocation failed, size = " 
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
    if (cudaMemcpyAsync(input_mem, &in_points, input_size, 
            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        gLogError << "ERROR: CUDA memory copy of input failed, size = " 
            << input_size << " bytes" << std::endl;
        return false;
    }

    // Run TensorRT inference
    void* bindings[] = {input_mem, keypoints_mem, features_mem, attention_mem};
    bool status = context->enqueueV2(bindings, stream, nullptr);
    if (!status) {
        gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    }

    //~ Copy predictions from output binding memory.
    // Copy keypoints
    auto keypoints_buffer = std::unique_ptr<float>{new float[keypoints_size]};
    if (cudaMemcpyAsync(keypoints_buffer.get(), keypoints_mem, keypoints_size, 
        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        gLogError << "ERROR: CUDA memory copy of keypoints failed, size = "
            << keypoints_size << " bytes" << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);

    // Copy features
    auto features_buffer = std::unique_ptr<float>{new float[features_size]};
    if (cudaMemcpyAsync(features_buffer.get(), features_mem, features_size, 
        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        gLogError << "ERROR: CUDA memory copy of features failed, size = "
            << features_size << " bytes" << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);

    // Copy attention
    auto attention_buffer = std::unique_ptr<float>{new float[attention_size]};
    if (cudaMemcpyAsync(attention_buffer.get(), attention_mem, attention_size, 
        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        gLogError << "ERROR: CUDA memory copy of attention failed, size = "
            << attention_size << " bytes" << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);

    // Write to output object
    out_points.attention = attention_buffer.get();
    out_points.keypoints = keypoints_buffer.get();
    out_points.features = features_buffer.get();


    // Free CUDA resources
    cudaFree(input_mem);
    cudaFree(keypoints_mem);
    cudaFree(features_mem);
    cudaFree(attention_mem);
    return true;
}

extern int MAX_KEYPOINTS;   // defined in main()
// Reads a float32 from the input file.
void ReadVariableFromBin(std::vector<float> &vect, 
                        std::string filename, const int dims=6){
    float f;
    std::ifstream f_in(filename, std::ios::binary);

    while( f_in.read(reinterpret_cast<char*>(&f), sizeof(float)) ){
        vect.push_back(f);
    }

    assertm( vect.size()%dims==0, "Input data should be a multiple of dims!" );
}

// non-max suppression
infer_output nms(infer_output input, int num_models=1){
    // num_models should be equal to batch size (usu 1)
    std::vector<int32_t> num_keypoints (num_models, 0); // num_models elements of value 0

    pointcloud_t xyz_nms(num_models, MAX_KEYPOINTS, 3);
    xyz_nms.setConstant(0.0);
    
    Eigen::Tensor<float, 2> attention_nms(num_models, MAX_KEYPOINTS);
    attention_nms.setConstant(0.0);

    for(int i=0; i<num_models; i++){
        // ! Find 50 nearest neighbors in each slice
        // nbrs = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(xyz[i, :, :])
        // distances, indices = nbrs.kneighbors(xyz[i, :, :])
    }


        knn_attention = attention[i, indices]
        outside_ball = distances > args.nms_radius
        knn_attention[outside_ball] = 0.0
        is_max = np.where(np.argmax(knn_attention, axis=1) == 0)[0]

        # Extract the top k features, filtering out weak responses
        attention_thresh = np.max(attention[i, :]) * args.min_response_ratio
        is_max_attention = [(attention[i, m], m) for m in is_max if attention[i, m] > attention_thresh]
        is_max_attention = sorted(is_max_attention, reverse=True)
        max_indices = [m[1] for m in is_max_attention]

        if len(max_indices) >= args.max_keypoints:
            max_indices = max_indices[:args.max_keypoints]
            num_keypoints[i] = len(max_indices)
        else:
            num_keypoints[i] = len(max_indices)  # Retrain original number of points
            max_indices = np.pad(max_indices, (0, args.max_keypoints - len(max_indices)), 'constant',
                                 constant_values=max_indices[0])

        xyz_nms[i, :, :] = xyz[i, max_indices, :]
        attention_nms[i, :] = attention[i, max_indices]

    return xyz_nms, attention_nms, num_keypoints
}