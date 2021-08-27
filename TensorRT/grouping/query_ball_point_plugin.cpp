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

// Overriding IPluginV2's virtual functions
AsciiChar const * QueryBallPointPlugin::getPluginType () const noexcept {
    return QUERYBALLPOINT_PLUGIN_NAME;
}
AsciiChar const * QueryBallPointPlugin::getPluginVersion () const noexcept {
    return QUERYBALLPOINT_PLUGIN_VERSION;
}
int32_t QueryBallPointPlugin::getNbOutputs () const noexcept {
    // get number of outputs:
    // b: batch size, m: npoints, nsample: external param (args.num_sample)
    // idx (b,m,nsample)
    // pts_cnt (b,m)
    return 2;
}

int32_t QueryBallPointPlugin::initialize () noexcept{
}
void QueryBallPointPlugin::terminate () noexcept{
}
size_t QueryBallPointPlugin::getSerializationSize () const noexcept{
}
void QueryBallPointPlugin::serialize (void *buffer) const noexcept{
}
void QueryBallPointPlugin::destroy () noexcept{
}
void QueryBallPointPlugin::setPluginNamespace (AsciiChar const *pluginNamespace) noexcept{
}
AsciiChar const * QueryBallPointPlugin::getPluginNamespace () const noexcept{
}

// Overriding IPluginV2Ext's virtual functions
nvinfer1::DataType QueryBallPointPlugin::getOutputDataType 
        (int32_t index, nvinfer1::DataType const *inputTypes, 
        int32_t nbInputs) const noexcept{
}

void QueryBallPointPlugin::attachToContext 
        (cudnnContext *, cublasContext *, IGpuAllocator *) noexcept{
}

void QueryBallPointPlugin::detachFromContext () noexcept{
}

// Overriding IPluginV2DynamicExt's virtual functions
DimsExprs QueryBallPointPlugin::getOutputDimensions
        (int32_t outputIndex, const DimsExprs *inputs, 
        int32_t nbInputs, IExprBuilder &exprBuilder) noexcept{
    // validate input args
    // QueryBallPoint has 2 inputs:
        // Attrs: radius (1), nsample (1)
        // Inputs: xyz1 (b,n,3), xyz2 (b,m,3)
        // Outputs: output1(b, m, nsample), output2(b,m)
    assertm(nbInputs==2, 
        "QueryBallPoint has 4 inputs:\ninput: xyz1 (b,n,3), xyz2 (b,m,3)"
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

bool QueryBallPointPlugin::supportsFormatCombination 
        (int32_t pos, const PluginTensorDesc *inOut, 
        int32_t nbInputs, int32_t nbOutputs) noexcept{
}

void QueryBallPointPlugin::configurePlugin 
        (const DynamicPluginTensorDesc *in, int32_t nbInputs,
        const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept{
}

size_t QueryBallPointPlugin::getWorkspaceSize
        (const PluginTensorDesc *inputs, int32_t nbInputs, 
        const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept{
}

int32_t QueryBallPointPlugin::enqueue 
        (const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, 
        const void *const *inputs, void *const *outputs, 
        void *workspace, cudaStream_t stream) noexcept{
}

int ClipPlugin::initialize() noexcept
{
    return 0;
}

int ClipPlugin::enqueue(int batchSize, const void* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept
{
    int status = -1;

    // Our plugin outputs only one tensor
    void* output = outputs[0];

    // Launch CUDA kernel wrapper and save its return value
    status = clipInference(stream, mInputVolume * batchSize, mClipMin, mClipMax, inputs[0], output);

    return status;
}

size_t ClipPlugin::getSerializationSize() const noexcept
{
    return 2 * sizeof(float);
}

void ClipPlugin::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    const char* a = d;

    writeToBuffer(d, mClipMin);
    writeToBuffer(d, mClipMax);

    assert(d == a + getSerializationSize());
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

void ClipPlugin::terminate() noexcept {}

void ClipPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* ClipPlugin::clone() const noexcept
{
    auto plugin = new ClipPlugin(mLayerName, mClipMin, mClipMax);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void ClipPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* ClipPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
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
