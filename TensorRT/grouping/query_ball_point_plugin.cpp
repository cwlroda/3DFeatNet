/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "grouping_plugin.h"
#include "NvInfer.h"
#include "grouping_kernel.h"
#include "noisy_assert.h"

#include <cassert>
#include <cstring>
#include <vector>

using namespace nvinfer1;

// Clip plugin specific constants
namespace
{
const char* QUERYBALLPOINT_PLUGIN_VERSION{"1"};
const char* QUERYBALLPOINT_PLUGIN_NAME{"QueryBallPointPlugin"};
const int32_t _NSAMPLE = 64;    // xref inference_tf2.py (args.num_samples)
} // namespace

// Static class fields initialization
PluginFieldCollection ClipPluginCreator::mFC{};
std::vector<PluginField> ClipPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(QueryBallPointPluginCreator);

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

// Define Constructors for each new Custom Layer
QueryBallPointPlugin::QueryBallPointPlugin(const std::string name,
    const float radius, const int32_t num_samples)
    : mLayerName(name), _radius(radius), _num_samples(num_samples)
{}

// Returns the name of the plugin; ie QueryBallPoint
AsciiChar const * QueryBallPointPlugin::getPluginType () const noexcept {
    return QUERYBALLPOINT_PLUGIN_NAME;
}

// Returns the name of version of the plugin, ie 1.0
AsciiChar const * QueryBallPointPlugin::getPluginVersion () const noexcept {
    return QUERYBALLPOINT_PLUGIN_VERSION;
}

// get number of outputs to the plugin:
// where {b: batch size, m: npoints, nsample: external param (args.num_sample)}
// return:
// idx (b,m,nsample)
// pts_cnt (b,m)
int32_t QueryBallPointPlugin::getNbOutputs () const noexcept { return 2; }

// Initialises plugin for inference, just return 0
int32_t QueryBallPointPlugin::initialize () noexcept{ return 0; }

// Release resources acquired during plugin layer initialization.
// This is called when the engine is destroyed.
void QueryBallPointPlugin::terminate () noexcept{}

// Returns the size of the serialization buffer.
// Seems to be what's needed to save the variables in this op,
// ie the class private variables.
size_t QueryBallPointPlugin::getSerializationSize () const noexcept{
    // float: _radius, int32_t: _num_samples
    return sizeof(float) + sizeof(int32_t);
}

/* Serialize the layer.
    Parameters:
        buffer - Pointer to a buffer to serialize data. 
        Size of buffer must be equal to value returned by 
        getSerializationSize()
*/
void QueryBallPointPlugin::serialize (void *buffer) const noexcept{
    char* d = static_cast<char*>(buffer);
    const char* a = d;  // start of mem buffer

    writeToBuffer(d, _radius);
    writeToBuffer(d, _num_samples);

    assertm(
        (d == a + getSerializationSize()),
        "Size of memory buffer in serialization must be equal to value returned by \
        `getSerializationSize`, sizeof(int32+float)"
    );
}

// This gets called when the network, builder or engine 
// containing this plugin is destroyed.
void QueryBallPointPlugin::destroy () noexcept{
    delete this;
}

// Set the namespace that this plugin object belongs to. 
// Ideally, all plugin objects from the same plugin library should have 
// the same namespace.
void QueryBallPointPlugin::setPluginNamespace(AsciiChar const *pluginNamespace) noexcept{
    this->mNamespace = pluginNamespace;
}

// Return the namespace of the plugin object
AsciiChar const * QueryBallPointPlugin::getPluginNamespace () const noexcept{
    return this->mNamespace.c_str();
}

// ~ Overriding IPluginV2Ext's virtual functions //

/*
Return the DataType of the plugin output at the requested index. 
The default behavior should be to return the type of the first input, 
or `DataType::kFLOAT` if the layer has no inputs. 

The returned data type must have a format that is supported by the plugin.
*/
nvinfer1::DataType QueryBallPointPlugin::getOutputDataType 
        (int32_t index, nvinfer1::DataType const *inputTypes, 
        int32_t nbInputs) const noexcept{
    assertm(nbInputs==2,
        "QueryBallPoint has 2 inputs:\ninput: xyz1 (b,n,3), xyz2 (b,m,3)"
    );
    assertm(index<2, "Only two outputs to QueryBallPoint.");
    // QueryBallPoint returns idx(b,n,nsample), pts_cnt (b,m)
    // both idx and pnts_cnt are int32.
    return DataType::kINT32;
}

/*
Attach the plugin object to an execution context and grant the plugin the access 
to some context resource.

Inputs:
    cudnn:	    The CUDNN context handle of the execution context
    cublas:	    The cublas context handle of the execution context
    allocator:  The allocator used by the execution context
*/
void QueryBallPointPlugin::attachToContext 
        (cudnnContext *, cublasContext *, IGpuAllocator *) noexcept{
    // not sure how to use this...
    // TODO get rid of this when safe
    printf("### Called QueryBallPointPlugin::attachToContext ###\n");
}

/*
Detach the plugin object from its execution context.
This function is called automatically for each plugin when a execution context is 
destroyed or the context resources are unassigned from the context.

If the plugin owns per-context resource, it can be released here.
*/
void QueryBallPointPlugin::detachFromContext () noexcept{
    // not sure how to use this...
    // TODO get rid of this when safe
    printf("### Called QueryBallPointPlugin::detachFromContext ###\n");
}

// ~ Overriding IPluginV2DynamicExt's virtual functions

/**/
DimsExprs QueryBallPointPlugin::getOutputDimensions
        (int32_t outputIndex, const DimsExprs *inputs, 
        int32_t nbInputs, IExprBuilder &exprBuilder) noexcept{
    // validate input args
    // QueryBallPoint has 2 inputs:
        // Attrs: radius (1), nsample (1)
        // Inputs: xyz1 (b,n,3), xyz2 (b,m,3)
        // Outputs: output1(b, m, nsample), output2(b,m)
    assertm(nbInputs==2, 
        "QueryBallPoint has 2 inputs:\ninput: xyz1 (b,n,3), xyz2 (b,m,3)"
    );
    assertm(
        inputs[0].nbDims==3,
        "xyz1 (input 1) is supposed to have final dimension 3."
    );
    assertm(
        inputs[1].nbDims==3,
        "xyz2 (input 2) is supposed to have final dimension 3."
    );
    assertm(
        outputIndex<2,
        "There are only two output tensors!"
    );
    int batchsize = inputs[1].d[0]->getConstantValue();
    int npoint = inputs[1].d[1]->getConstantValue();

    // Input2 needs to have certain dimensions (batchsize, npoint, 3)
    DimsExprs output;

    if (outputIndex==0){
        output.d[0] = exprBuilder.constant(batchsize);
        output.d[1] = exprBuilder.constant(npoint);
        output.d[2] = exprBuilder.constant(_num_samples);
    } else {
        output.d[0] = exprBuilder.constant(batchsize);
        output.d[1] = exprBuilder.constant(npoint);
    }

    return output;
}

/*
Return true if plugin supports the format and datatype for the input/output
indexed by pos.

Pos must be within (nbInputs+nbOutputs-1).
*/
bool QueryBallPointPlugin::supportsFormatCombination 
        (int32_t pos, const PluginTensorDesc *inOut, 
        int32_t nbInputs, int32_t nbOutputs) noexcept{

    // Network supports:
    // FP32 NCHW for both inputs.
    // int32 outputs.

    assertm( (0<=pos && pos<4), 
        "Position should be between 0-3 (2 inputs, 2 outputs)"
        );

    // Check if outputs are int32 
    if (pos >= 2){
        return inOut[pos].type == DataType::kINT32;         // outputs int32
    } else {
        return inOut[pos].type == DataType::kFLOAT &&       // float32
            inOut[pos].format == TensorFormat::kLINEAR;     // NCHW
    }

    return false;
}

/*
Used to configure any plugins required by this given plugin (none in this case)
*/
void QueryBallPointPlugin::configurePlugin 
        (const DynamicPluginTensorDesc *in, int32_t nbInputs,
        const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept{
    // TODO: Get rid of this when figured out safe.
    printf("### Called QueryBallPointPlugin::configurePlugin ###\n");
}

/*
Find the workspace size required by the layer.
This function is called after the plugin is configured, and possibly during execution. 
The result should be a sufficient workspace size to deal with inputs and outputs of 
the given size or any smaller problem.

As this plugin computes data in-place, we don't need this.
*/
size_t QueryBallPointPlugin::getWorkspaceSize
        (const PluginTensorDesc *inputs, int32_t nbInputs, 
        const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept{
    // TODO: Figure this out
    return 0;
}

/*
Execute the layer.
Inputs:
    inputDesc	how to interpret the memory for the input tensors.
    outputDesc	how to interpret the memory for the output tensors.
    inputs	The memory for the input tensors.
    outputs	The memory for the output tensors.
    workspace	Workspace for execution.
    stream	The stream in which to execute the kernels.

Returns
    0 for success, else non-zero (which will cause engine termination).
*/
int32_t QueryBallPointPlugin::enqueue 
        (const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, 
        const void *const *inputs, void *const *outputs, 
        void *workspace, cudaStream_t stream) noexcept{
    
    // todo after lunch
    

    int batchsize = inputs[1].d[0]->getConstantValue();
    int npoint = inputs[1].d[1]->getConstantValue();


    // Launch inference kernel
    queryBallPointLauncher(b, n, m, _radius, _num_samples, 
                    xyz1, xyz2, idx, pts_cnt
                    );


    const Tensor& xyz1_tensor = context->input(0);
    OP_REQUIRES(context, xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 3) xyz1 shape."));
    int b = xyz1_tensor.shape().dim_size(0);
    int n = xyz1_tensor.shape().dim_size(1);

    const Tensor& xyz2_tensor = context->input(1);
    OP_REQUIRES(context, xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size, npoint, 3) xyz2 shape."));
    int m = xyz2_tensor.shape().dim_size(1);

    Tensor *idx_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,nsample_}, &idx_tensor));
    Tensor *pts_cnt_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m}, &pts_cnt_tensor));

    auto xyz1_flat = xyz1_tensor.flat<float>();
    const float *xyz1 = &(xyz1_flat(0));
    auto xyz2_flat = xyz2_tensor.flat<float>();
    const float *xyz2 = &(xyz2_flat(0));
    auto idx_flat = idx_tensor->flat<int>();
    int *idx = &(idx_flat(0));
    auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
    int *pts_cnt = &(pts_cnt_flat(0));
    queryBallPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx,pts_cnt);


    return 0;
}

// ###################################################### //
// ###################################################### //
// ###################################################### //
// ###################################################### //
// ###################################################### //
// ###################################################### //

int ClipPlugin::enqueue(int batchSize, const void* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept
{
    int status = -1;

    // Our plugin outputs only one tensor
    void* output = outputs[0];

    // Launch CUDA kernel wrapper and save its return value
    status = clipInference(stream, mInputVolume * batchSize, mClipMin, mClipMax, inputs[0], output);

    return status;
}

void ClipPlugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs,
    DataType type, PluginFormat format, int) noexcept
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(type == DataType::kFLOAT);
    assert(format == PluginFormat::kLINEAR);

    // Fetch volume for future enqueue() operations
    size_t volume = 1;
    for (int i = 0; i < inputs->nbDims; i++)
    {
        volume *= inputs->d[i];
    }
    mInputVolume = volume;
}

bool ClipPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    // This plugin only supports ordinary floats, and NCHW input format
    if (type == DataType::kFLOAT && format == PluginFormat::kLINEAR)
        return true;
    else
        return false;
}

IPluginV2* ClipPlugin::clone() const noexcept
{
    auto plugin = new ClipPlugin(mLayerName, mClipMin, mClipMax);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}


ClipPluginCreator::ClipPluginCreator()
{
    // Describe ClipPlugin's required PluginField arguments
    mPluginAttributes.emplace_back(PluginField("clipMin", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("clipMax", nullptr, PluginFieldType::kFLOAT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ClipPluginCreator::getPluginName() const noexcept
{
    return CLIP_PLUGIN_NAME;
}

const char* ClipPluginCreator::getPluginVersion() const noexcept
{
    return CLIP_PLUGIN_VERSION;
}

const PluginFieldCollection* ClipPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* ClipPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    float clipMin, clipMax;
    const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    assert(fc->nbFields == 2);
    for (int i = 0; i < fc->nbFields; i++)
    {
        if (strcmp(fields[i].name, "clipMin") == 0)
        {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            clipMin = *(static_cast<const float*>(fields[i].data));
        }
        else if (strcmp(fields[i].name, "clipMax") == 0)
        {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            clipMax = *(static_cast<const float*>(fields[i].data));
        }
    }
    return new ClipPlugin(name, clipMin, clipMax);
}

IPluginV2* ClipPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call ClipPlugin::destroy()
    return new ClipPlugin(name, serialData, serialLength);
}

void ClipPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* ClipPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
