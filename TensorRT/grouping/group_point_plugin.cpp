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
* Defines the functions used in QueryBallPoint.
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
PluginFieldCollection QueryBallPointPluginCreator::mFC{};
std::vector<PluginField> QueryBallPointPluginCreator::mPluginAttributes;

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

// #################################################################################### //
/*                   Function implementations for QueryBallPointPlugin                  */
// #################################################################################### //

// Default constructor for QueryBallPoint (ie)
QueryBallPointPlugin::QueryBallPointPlugin(const std::string name,
    const float radius, const int32_t num_samples)
    : mLayerName(name), _radius(radius), _num_samples(num_samples)
{}

QueryBallPointPlugin::QueryBallPointPlugin
    (const std::string name, const void* data, size_t length){

    assertm(length==getSerializationSize(), "Serialised length must be \
    sizeof(int32+float): radius + num_samples.");

    const char* d = static_cast<const char*>(data);
    const char* a = d;

    _radius = readFromBuffer<float>(d);
    _num_samples = readFromBuffer<int32_t>(d);

    assert(d == (a + length));
}


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

IPluginV2DynamicExt* QueryBallPointPlugin::clone () const noexcept {
    auto plugin = new QueryBallPointPlugin(mLayerName, _radius, _num_samples);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

/*
Does what it says on the tin.
*/
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
    
    assertm((inputDesc[0].dims.nbDims==3 && inputDesc[0].dims.d[2]==3), 
        "QueryBallPoint expects (batch_size, ndataset, 3) xyz1 shape."
    );
    assertm((inputDesc[1].dims.nbDims==3 && inputDesc[1].dims.d[2]==3), 
        "QueryBallPoint expects (batch_size, npoint, 3) xyz2 shape."
    );
    // declare 'inputs' as pointer to const pointer to const void
    // declare 'outputs' as as pointer to const pointer to void
    int b = inputDesc[0].dims.d[0]; // batch size
    int n = inputDesc[0].dims.d[1];
    int m = inputDesc[1].dims.d[1];

    // handlers for input. Should be flattened already since already in memory.
    const float* xyz1 = static_cast<const float*>(inputs[0]);
    const float* xyz2 = static_cast<const float*>(inputs[1]);

    // handlers for output
    int32_t* idx = static_cast<int32_t*>(outputs[0]);
    int32_t* pts_cnt = static_cast<int32_t*>(outputs[1]);

    // Launch inference kernel
    queryBallPointLauncher(b, n, m, _radius, _num_samples, 
                    xyz1, xyz2, idx, pts_cnt
                    );

    assertm( (outputDesc[0].dims.nbDims==3 && outputDesc[0].dims.d[2]==_num_samples),
        "Output idx must have shape (batch_size, npoint, num_samples)"
    );
    assertm( (outputDesc[1].dims.nbDims==2) ,
        "Output pts_cnt must have shape (batch_size, npoint)"
    );

    return 0;
}
// #################################################################################### //

// #################################################################################### //
/*                       Functions for QueryBallPointPluginCreator                      */
// #################################################################################### //

QueryBallPointPluginCreator::QueryBallPointPluginCreator(){
    // Describe QBPPlugin's required PluginField args (radius, num_samples)
    mPluginAttributes.emplace_back(
            PluginField("_radius", nullptr, PluginFieldType::kFLOAT32, 1));

    mPluginAttributes.emplace_back(
            PluginField("_num_samples", nullptr, PluginFieldType::kINT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* QueryBallPointPluginCreator::getPluginName() const noexcept{
    return QUERYBALLPOINT_PLUGIN_NAME;
}

const char* QueryBallPointPluginCreator::getPluginVersion() const noexcept{
    return QUERYBALLPOINT_PLUGIN_VERSION;
}

const PluginFieldCollection* QueryBallPointPluginCreator::getFieldNames() noexcept{
    return &mFC;
}

// Creates a new instance of QueryBallPointPlugin
IPluginV2* QueryBallPointPluginCreator::createPlugin
    (const char* name, const PluginFieldCollection* fc) noexcept{

    float radius;
    int32_t num_samples;
    const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    assertm((fc->nbFields == 2), "Input PluginFiledCollection must only have 2 fields.");
    
    for (int i = 0; i < fc->nbFields; i++) {
        if (strcmp(fields[i].name, "_radius") == 0)
        {
            assertm((fields[i].type == PluginFieldType::kFLOAT32), 
                "Provided radius parameter must have datatype float32");
            radius = *(static_cast<const float*>(fields[i].data));
        }
        else if (strcmp(fields[i].name, "_num_samples") == 0)
        {
            assertm((fields[i].type == PluginFieldType::kINT32), 
                "Provided num_samples parameter must have datatype int32");
            num_samples = *(static_cast<const float*>(fields[i].data));
        }
    }

    return new QueryBallPointPlugin(name, radius, num_samples);
}

IPluginV2* QueryBallPointPluginCreator::deserializePlugin
    (const char* name, const void* serialData, size_t serialLength) noexcept{
    // This object will be deleted when the network is destroyed, which will
    // call ClipPlugin::destroy()
    return new QueryBallPointPlugin(name, serialData, serialLength);
}

void QueryBallPointPluginCreator::setPluginNamespace
    (const char* pluginNamespace) noexcept{
    
    mNamespace = pluginNamespace;
}

const char* QueryBallPointPluginCreator::getPluginNamespace() const noexcept{
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