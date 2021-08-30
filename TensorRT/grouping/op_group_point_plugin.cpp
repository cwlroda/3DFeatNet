/*
Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

* By Tianyi
* Defines the functions used in GroupPoint.
*/

#include "NvInfer.h"
#include "grouping_plugin.h"
#include "grouping_kernel.h"
#include "noisy_assert.h"

#include <cassert>
#include <cstring>
#include <vector>

using namespace nvinfer1;

// Clip plugin specific constants
namespace
{
const char* GROUPPOINT_PLUGIN_VERSION{"1"};
const char* GROUPPOINT_PLUGIN_NAME{"GroupPoint"};
} // namespace

// Static class fields initialization
PluginFieldCollection GroupPointPluginCreator::mFC{};
std::vector<PluginField> GroupPointPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GroupPointPluginCreator);

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

// #################################################################################### //
/*                     Function implementations for GroupPointPlugin                    */
// #################################################################################### //

// Default constructor for GroupPoint (nothing to init)
GroupPointPlugin::GroupPointPlugin(const std::string name){}

// Secondary constructor for GroupPoint, but similarly, nothing to init
GroupPointPlugin::GroupPointPlugin
    (const std::string name, const void* data, size_t length){}

// Returns the name of the plugin; ie GroupPoint
AsciiChar const * GroupPointPlugin::getPluginType () const noexcept {
    return GROUPPOINT_PLUGIN_NAME;
}

// Returns the name of version of the plugin, ie 1.0
AsciiChar const * GroupPointPlugin::getPluginVersion () const noexcept {
    return GROUPPOINT_PLUGIN_VERSION;
}

// get number of outputs to the plugin:
// where {b: batch size, m: npoints, 
//    nsample (passed in beforehand), c: channels}
// return:
// output: out (b,m,nsample,c)
int32_t GroupPointPlugin::getNbOutputs () const noexcept { return 1; }

// Initialises plugin for inference, just return 0
int32_t GroupPointPlugin::initialize () noexcept{ return 0; }

// Release resources acquired during plugin layer initialization.
// This is called when the engine is destroyed.
void GroupPointPlugin::terminate () noexcept{}

// Returns the size of the serialization buffer.
// Seems to be what's needed to save the variables in this op,
// ie the class private variables.
// In the case of GroupPoint there is nothing to serialize.
size_t GroupPointPlugin::getSerializationSize () const noexcept{ return 0; }

/* Serialize the layer.
    Parameters:
        buffer - Pointer to a buffer to serialize data. 
        Size of buffer must be equal to value returned by 
        getSerializationSize()
*/
void GroupPointPlugin::serialize (void *buffer) const noexcept{}

// This gets called when the network, builder or engine 
// containing this plugin is destroyed.
void GroupPointPlugin::destroy () noexcept{ delete this; }

// Set the namespace that this plugin object belongs to. 
// Ideally, all plugin objects from the same plugin library should have 
// the same namespace.
void GroupPointPlugin::setPluginNamespace(AsciiChar const *pluginNamespace) noexcept{
    this->mNamespace = pluginNamespace;
}

// Return the namespace of the plugin object
AsciiChar const * GroupPointPlugin::getPluginNamespace () const noexcept{
    return this->mNamespace.c_str();
}

// ~ Overriding IPluginV2Ext's virtual functions //

/*
Return the DataType of the plugin output at the requested index. 
The default behavior should be to return the type of the first input, 
or `DataType::kFLOAT` if the layer has no inputs. 

The returned data type must have a format that is supported by the plugin.
*/
nvinfer1::DataType GroupPointPlugin::getOutputDataType 
        (int32_t index, nvinfer1::DataType const *inputTypes, 
        int32_t nbInputs) const noexcept{
    assertm(nbInputs==2,
        "GroupPoint has 2 inputs:\ninput: points (b,n,3), idx (b,m,nsample)"
    );
    assertm(index==0, "Only one output to GroupPoint.");
    // GroupPoint returns (batch_size, npoint, nsample, channel)
    // a float32 array
    return DataType::kFLOAT;
}

/*
Attach the plugin object to an execution context and grant the plugin the access 
to some context resource.

Inputs:
    cudnn:	    The CUDNN context handle of the execution context
    cublas:	    The cublas context handle of the execution context
    allocator:  The allocator used by the execution context
*/
void GroupPointPlugin::attachToContext 
        (cudnnContext *, cublasContext *, IGpuAllocator *) noexcept{
    // not sure how to use this...
    // TODO get rid of this when safe
    printf("### Called GroupPointPlugin::attachToContext ###\n");
}

/*
Detach the plugin object from its execution context.
This function is called automatically for each plugin when a execution context is 
destroyed or the context resources are unassigned from the context.

If the plugin owns per-context resource, it can be released here.
*/
void GroupPointPlugin::detachFromContext () noexcept{
    // not sure how to use this...
    // TODO get rid of this when safe
    printf("### Called GroupPointPlugin::detachFromContext ###\n");
}

// ~ Overriding IPluginV2DynamicExt's virtual functions

IPluginV2DynamicExt* GroupPointPlugin::clone () const noexcept {
    auto plugin = new GroupPointPlugin(mLayerName);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

/*
Does what it says on the tin.
*/
DimsExprs GroupPointPlugin::getOutputDimensions
        (int32_t outputIndex, const DimsExprs *inputs, 
        int32_t nbInputs, IExprBuilder &exprBuilder) noexcept{
    // validate input args

    // Input:
    //     points: (batch_size, ndataset, channel) float32 arr, points to sample from
    //     idx: (batch_size, npoint, nsample) int32 arr, indices to points
    // Output:
    //     out: (batch_size, npoint, nsample, channel) 
    //     float32 array, values sampled from points

    assertm(nbInputs==2, 
        "GroupPoint has 2 inputs:\ninput: points (b,n,channel), idx (b,m,nsample)"
    );
    assertm(
        (inputs[0].nbDims==3),
        "points (input 1) has 3 dimensions."
    );
    assertm(
        (inputs[1].nbDims==3),
        "idx (input 2) is supposed to have final dimension 3."
    );
    assertm(
        outputIndex==0,
        "There is only one output tensor from GroupPoint."
    );
    // int batchsize = inputs[1].d[0]->getConstantValue();
    // int npoint = inputs[1].d[1]->getConstantValue();
    // int nsample = inputs[1].d[2]->getConstantValue();
    // int nchannel = inputs[0].d[2]->getConstantValue();

    // Input needs to have certain dimensions (batch_size, npoint, nsample, channel) 
    DimsExprs output(inputs[1]);
    output.nbDims = 4;
    // output.d[0] = inputs[1].d[0]->getConstantValue();    // batch size
    // output.d[1] = inputs[1].d[1]->getConstantValue();    // npoint
    // output.d[2] = inputs[1].d[2]->getConstantValue();    // nsample
    output.d[3] = exprBuilder.constant(   
        inputs[0].d[2]->getConstantValue()        // nchannel
    );

    // printf("Output dimensions of GroupPoint: %d\n", output.nbDims);
    // for(int i=0; i<output.nbDims; i++){
    //     printf("%d\t", output.d[i]->getConstantValue());
    // }
    // printf("\n");

    return output;
}

/*
Return true if plugin supports the format and datatype for the input/output
indexed by pos.

Pos must be within (nbInputs+nbOutputs-1).
*/
bool GroupPointPlugin::supportsFormatCombination 
        (int32_t pos, const PluginTensorDesc *inOut, 
        int32_t nbInputs, int32_t nbOutputs) noexcept{

    // Network supports:
    // FP32 NHWC for both inputs.
    // float32 outputs.

    assertm( (0<=pos && pos<3), 
        "Position should be between 0-2 (2 inputs, 1 output)"
        );

    // Check if outputs are int32 
    switch (pos)
    {
    case 0:
        return inOut[pos].type == DataType::kFLOAT &&
            inOut[pos].format == TensorFormat::kHWC; // points: float array in NHWC
    case 1:
        return inOut[pos].type == DataType::kINT32; // idx, int32 array
    case 2:
        return inOut[pos].type == DataType::kFLOAT; // output is a float array
    default:
        return false;
    }
}

/*
Used to configure any plugins required by this given plugin (none in this case)
*/
void GroupPointPlugin::configurePlugin 
        (const DynamicPluginTensorDesc *in, int32_t nbInputs,
        const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept{
    // TODO: Get rid of this when figured out safe.
    printf("### Called GroupPointPlugin::configurePlugin ###\n");
}

/*
Find the workspace size required by the layer.
This function is called after the plugin is configured, and possibly during execution. 
The result should be a sufficient workspace size to deal with inputs and outputs of 
the given size or any smaller problem.

As this plugin computes data in-place, we don't need this.
*/
size_t GroupPointPlugin::getWorkspaceSize
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
int32_t GroupPointPlugin::enqueue 
        (const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, 
        const void *const *inputs, void *const *outputs, 
        void *workspace, cudaStream_t stream) noexcept{
    /*
    Input:
        points: (batch_size, ndataset, channel) float32 array, points to sample from
        idx: (batch_size, npoint, nsample) int32 array, indices to points
    Output:
        out: (batch_size, npoint, nsample, channel) float32 array, values sampled from points
    */
    assertm((inputDesc[0].dims.nbDims==3), 
        "GroupPoint expects (batch_size, ndataset, channel) points shape."
    );
    assertm((inputDesc[1].dims.nbDims==3), 
        "GroupPoint expects (batch_size, npoint, nsample) idx shape."
    );
    // 'inputs' is a pointer to const pointer to const void
    // 'outputs' is a pointer to const pointer to void
    int b = inputDesc[0].dims.d[0];         // batch size
    int n = inputDesc[0].dims.d[1];  // ndataset
    int c = inputDesc[0].dims.d[2];  // nchannel
    int m = inputDesc[1].dims.d[1];    // npoint
    int nsample = inputDesc[1].dims.d[2];   // nsample

    // handlers for input. Should be flattened already since already in memory.
    const float* points = static_cast<const float*>(inputs[0]);
    const int32_t* idx = static_cast<const int32_t*>(inputs[1]);

    // handlers for output
    float* out = static_cast<float*>(outputs[0]);

    // Launch inference kernel
    groupPointLauncher(b,n,c,m,nsample,points,idx,out);

    assertm( (outputDesc[0].dims.nbDims==4),
        "Output idx must have shape (batch_size, npoint, nsample, channel)"
    );
    return 0;
}
// #################################################################################### //

// #################################################################################### //
/*                         Functions for GroupPointPluginCreator                        */
// #################################################################################### //

GroupPointPluginCreator::GroupPointPluginCreator(){
    // Describe GroupPointPlugin's required PluginField args. 
    // But GroupPointPlugin has no args, so it's blank.
}

const char* GroupPointPluginCreator::getPluginName() const noexcept{
    return GROUPPOINT_PLUGIN_NAME;
}

const char* GroupPointPluginCreator::getPluginVersion() const noexcept{
    return GROUPPOINT_PLUGIN_VERSION;
}

const PluginFieldCollection* GroupPointPluginCreator::getFieldNames() noexcept{
    return &mFC;
}

// Creates a new instance of GroupPointPlugin
IPluginV2* GroupPointPluginCreator::createPlugin
    (const char* name, const PluginFieldCollection* fc) noexcept{
    
    // Unused variable, so it's commented out for now.
    // const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    // assertm((fc->nbFields == 0), "Input PluginFiledCollection has 0 fields.");
    
    return new GroupPointPlugin(name);
}

IPluginV2* GroupPointPluginCreator::deserializePlugin
    (const char* name, const void* serialData, size_t serialLength) noexcept{
    // This object will be deleted when the network is destroyed, which will
    // call ClipPlugin::destroy()
    return new GroupPointPlugin(name, serialData, serialLength);
}

void GroupPointPluginCreator::setPluginNamespace
    (const char* pluginNamespace) noexcept{
    
    mNamespace = pluginNamespace;
}

const char* GroupPointPluginCreator::getPluginNamespace() const noexcept{
    return mNamespace.c_str();
}

// #################################################################################### //


// Unimplemented IPluginV2 functions, keep for reference
/*
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
*/