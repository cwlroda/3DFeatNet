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
* Defines the functions used in SignOp.
*/

#include "NvInfer.h"
#include "sign_op_plugin.h"
#include "sign_op_kernel.h"
#include "noisy_assert.h"

#include <cassert>
#include <cstring>
#include <vector>

using namespace nvinfer1;

// Clip plugin specific constants
namespace
{
const char* SIGN_OP_PLUGIN_VERSION{"1"};
const char* SIGN_OP_PLUGIN_NAME{"Sign"};
} // namespace

// Static class fields initialization
PluginFieldCollection SignOpPluginCreator::mFC{};
std::vector<PluginField> SignOpPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(SignOpPluginCreator);

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
/*                       Function implementations for SignOpPlugin                      */
// #################################################################################### //

// Default constructor for SignOp. No vars to implement.
SignOpPlugin::SignOpPlugin(const std::string name) {}

SignOpPlugin::SignOpPlugin
    (const std::string name, const void* data, size_t length){}

// Returns the name of the plugin; ie SignOp
AsciiChar const * SignOpPlugin::getPluginType () const noexcept {
    return SIGN_OP_PLUGIN_NAME;
}

// Returns the name of version of the plugin, ie 1.0
AsciiChar const * SignOpPlugin::getPluginVersion () const noexcept {
    return SIGN_OP_PLUGIN_VERSION;
}

// get number of outputs to the plugin
int32_t SignOpPlugin::getNbOutputs () const noexcept { return 1; }

// Initialises plugin for inference, just return 0
int32_t SignOpPlugin::initialize () noexcept{ return 0; }

// Release resources acquired during plugin layer initialization.
// This is called when the engine is destroyed.
void SignOpPlugin::terminate () noexcept{}

// Returns the size of the serialization buffer.
// Seems to be what's needed to save the variables in this op,
// ie the class private variables.
size_t SignOpPlugin::getSerializationSize () const noexcept{ return 0; }

/* Serialize the layer.
    Parameters:
        buffer - Pointer to a buffer to serialize data. 
        Size of buffer must be equal to value returned by 
        getSerializationSize()
*/
void SignOpPlugin::serialize (void *buffer) const noexcept{}

// This gets called when the network, builder or engine 
// containing this plugin is destroyed.
void SignOpPlugin::destroy () noexcept{ delete this; }

// Set the namespace that this plugin object belongs to. 
// Ideally, all plugin objects from the same plugin library should have 
// the same namespace.
void SignOpPlugin::setPluginNamespace(AsciiChar const *pluginNamespace) noexcept{
    this->mNamespace = pluginNamespace;
}

// Return the namespace of the plugin object
AsciiChar const * SignOpPlugin::getPluginNamespace () const noexcept{
    return this->mNamespace.c_str();
}

// ~ Overriding IPluginV2Ext's virtual functions //

/*
Return the DataType of the plugin output at the requested index. 
The default behavior should be to return the type of the first input, 
or `DataType::kFLOAT` if the layer has no inputs. 

The returned data type must have a format that is supported by the plugin.
*/
nvinfer1::DataType SignOpPlugin::getOutputDataType 
        (int32_t index, nvinfer1::DataType const *inputTypes, 
        int32_t nbInputs) const noexcept{
    assertm(nbInputs==1, "SignOp has 1 input.");
    assertm(index<1, "Only one output to SignOp.");
    // SignOp returns idx(b,n,nsample), pts_cnt (b,m)
    // both idx and pnts_cnt are int32.
    return inputTypes[0];   // same output data type as input
}

/*
Attach the plugin object to an execution context and grant the plugin the access 
to some context resource.

Inputs:
    cudnn:	    The CUDNN context handle of the execution context
    cublas:	    The cublas context handle of the execution context
    allocator:  The allocator used by the execution context
*/
void SignOpPlugin::attachToContext 
        (cudnnContext *, cublasContext *, IGpuAllocator *) noexcept{
    // not sure how to use this...
    // TODO get rid of this when safe
    printf("### Called SignOpPlugin::attachToContext ###\n");
}

/*
Detach the plugin object from its execution context.
This function is called automatically for each plugin when a execution context is 
destroyed or the context resources are unassigned from the context.

If the plugin owns per-context resource, it can be released here.
*/
void SignOpPlugin::detachFromContext () noexcept{
    // not sure how to use this...
    // TODO get rid of this when safe
    printf("### Called SignOpPlugin::detachFromContext ###\n");
}

// ~ Overriding IPluginV2DynamicExt's virtual functions

IPluginV2DynamicExt* SignOpPlugin::clone () const noexcept {
    auto plugin = new SignOpPlugin(mLayerName);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

/*
Does what it says on the tin.
*/
DimsExprs SignOpPlugin::getOutputDimensions
        (int32_t outputIndex, const DimsExprs *inputs, 
        int32_t nbInputs, IExprBuilder &exprBuilder) noexcept{
    // validate input args
    assertm(nbInputs==1, "SignOp has 1 input.");
    assertm(outputIndex<1, "Only one output to SignOp.");

    DimsExprs output(inputs[0]);        // Same dimension as output
    return output;
}

/*
Return true if plugin supports the format and datatype for the input/output
indexed by pos.

Pos must be within (nbInputs+nbOutputs-1).
*/
bool SignOpPlugin::supportsFormatCombination 
        (int32_t pos, const PluginTensorDesc *inOut, 
        int32_t nbInputs, int32_t nbOutputs) noexcept{

    // Network supports: anything so long the output is same dtype as input

    /*
    printf("Checking Sign inputs and outputs:\n");
    for(int i=0; i<nbInputs; i++){
        printf("Input %d has type %d and format %d.\n", i, inOut[i].type, inOut[i].format);
    }
    for(int i=nbInputs; i<nbInputs+nbOutputs; i++){
        printf("Output %d has type %d and format %d.\n", i-nbInputs, inOut[i].type, inOut[i].format);
    }
    */

    assertm( (0<=pos && pos<2), 
        "Position should be between 0-1 (1 inputs, 1 outputs)"
        );

    // Check if output is same type as input or unknown
    return (inOut[0].type == inOut[1].type) || (int32_t)inOut[1].type==-1;
}

/*
Used to configure any plugins required by this given plugin (none in this case)
*/
void SignOpPlugin::configurePlugin 
        (const DynamicPluginTensorDesc *in, int32_t nbInputs,
        const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept{
    // TODO: Get rid of this when figured out safe.
    printf("### Called SignOpPlugin::configurePlugin ###\n");
}

/*
Find the workspace size required by the layer.
This function is called after the plugin is configured, and possibly during execution. 
The result should be a sufficient workspace size to deal with inputs and outputs of 
the given size or any smaller problem.

As this plugin computes data in-place, we don't need this.
*/
size_t SignOpPlugin::getWorkspaceSize
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
int32_t SignOpPlugin::enqueue 
        (const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, 
        const void *const *inputs, void *const *outputs, 
        void *workspace, cudaStream_t stream) noexcept{
    
    int n;
    assertm( (inputDesc[0].type==DataType::kINT32 || inputDesc[0].type==DataType::kFLOAT),
        "Sign Op only supports int32 and float32 implementation!"
    );

    // get handlers to input and output tensors and launch inference kernel
    switch (inputDesc[0].type){
        case DataType::kINT32: {
            n = sizeof(inputs) / sizeof(int32_t);
            const int32_t* in = static_cast<const int32_t*>(inputs[0]);
            int32_t* out = static_cast<int32_t*>(outputs[0]);
            signOpLauncher(n, in, out);
            break;
        }
        case DataType::kFLOAT: {
            n = sizeof(inputs) / sizeof(float);
            const float* in = static_cast<const float*>(inputs[0]);
            float* out = static_cast<float*>(outputs[0]);
            signOpLauncher(n, in, out);
            break;
        }
        default: {
            // Noisily refuse to handle the other datatypes
            printf("Invalid input data type: %d\n", (int32_t)inputDesc[0].type);
        }
    }
    
    return 0;
}
// #################################################################################### //

// #################################################################################### //
/*                       Functions for SignOpPluginCreator                      */
// #################################################################################### //

SignOpPluginCreator::SignOpPluginCreator(){}

const char* SignOpPluginCreator::getPluginName() const noexcept{
    return SIGN_OP_PLUGIN_NAME;
}

const char* SignOpPluginCreator::getPluginVersion() const noexcept{
    return SIGN_OP_PLUGIN_VERSION;
}

const PluginFieldCollection* SignOpPluginCreator::getFieldNames() noexcept{
    return &mFC;
}

// Creates a new instance of SignOpPlugin
IPluginV2* SignOpPluginCreator::createPlugin
    (const char* name, const PluginFieldCollection* fc) noexcept{
    return new SignOpPlugin(name);
}

IPluginV2* SignOpPluginCreator::deserializePlugin
    (const char* name, const void* serialData, size_t serialLength) noexcept{
    // This object will be deleted when the network is destroyed, which will
    // call ClipPlugin::destroy()
    return new SignOpPlugin(name, serialData, serialLength);
}

void SignOpPluginCreator::setPluginNamespace
    (const char* pluginNamespace) noexcept{
    
    mNamespace = pluginNamespace;
}

const char* SignOpPluginCreator::getPluginNamespace() const noexcept{
    return mNamespace.c_str();
}

// #################################################################################### //