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
 * 
 * Template by NVIDIA, developed by Tianyi
 */

#ifndef GROUPING_PLUGIN_H
#define GROUPING_PLUGIN_H

#include "NvInferPlugin.h"
#include <string>
#include <vector>

using namespace nvinfer1;

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class QueryBallPointPlugin : public IPluginV2DynamicExt
{
public:
    // Constructor
    QueryBallPointPlugin(const std::string name, 
        const float radius, const int32_t num_samples);

    // doesn't make sense to call the constructor w/o args, so delete default constructor
    QueryBallPointPlugin() = delete;

    // override IPluginV2DynamicExt virtual functions
    DimsExprs getOutputDimensions (int32_t outputIndex, const DimsExprs *inputs, 
                    int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override;
    
    bool supportsFormatCombination (int32_t pos, const PluginTensorDesc *inOut, 
                    int32_t nbInputs, int32_t nbOutputs) noexcept override;
    
    void configurePlugin (const DynamicPluginTensorDesc *in, int32_t nbInputs,
                    const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize (const PluginTensorDesc *inputs, int32_t nbInputs, 
                    const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue (const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, 
                    const void *const *inputs, void *const *outputs, 
                    void *workspace, cudaStream_t stream) noexcept override;

    // override IPluginV2Ext's virtual functions
    nvinfer1::DataType getOutputDataType (int32_t index, nvinfer1::DataType const *inputTypes, 
                    int32_t nbInputs) const noexcept override;

    void attachToContext (cudnnContext *, cublasContext *, IGpuAllocator *) noexcept override;
    
    void detachFromContext () noexcept override;

    // override IPluginV2's virtual functions
    AsciiChar const * getPluginType () const noexcept override;
    
    AsciiChar const * getPluginVersion () const noexcept override;
    
    int32_t getNbOutputs () const noexcept override;
    
    int32_t initialize () noexcept override;
    
    void terminate () noexcept override;
    
    size_t getSerializationSize () const noexcept override;
    
    void serialize (void *buffer) const noexcept override;
    
    void destroy () noexcept override;
    
    void setPluginNamespace (AsciiChar const *pluginNamespace) noexcept override;
    
    AsciiChar const * getPluginNamespace () const noexcept override;

private:
    const std::string mLayerName;   // Name of the custom layer onject
    const float _radius;
    const int32_t _num_samples;
};

class GroupPointPlugin : public IPluginV2DynamicExt
{
    // Constructor
    GroupPointPlugin(const std::string name);

    // doesn't make sense to call the constructor w/o args, so delete default constructor
    GroupPointPlugin() = delete;

    // override IPluginV2DynamicExt virtual functions
    DimsExprs getOutputDimensions (int32_t outputIndex, const DimsExprs *inputs, 
                    int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override;
    
    bool supportsFormatCombination (int32_t pos, const PluginTensorDesc *inOut, 
                    int32_t nbInputs, int32_t nbOutputs) noexcept override;
    
    void configurePlugin (const DynamicPluginTensorDesc *in, int32_t nbInputs,
                    const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize (const PluginTensorDesc *inputs, int32_t nbInputs, 
                    const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue (const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, 
                    const void *const *inputs, void *const *outputs, 
                    void *workspace, cudaStream_t stream) noexcept override;

    // override IPluginV2Ext's virtual functions
    nvinfer1::DataType getOutputDataType (int32_t index, nvinfer1::DataType const *inputTypes, 
                    int32_t nbInputs) const noexcept override;

    void attachToContext (cudnnContext *, cublasContext *, IGpuAllocator *) noexcept override;
    
    void detachFromContext () noexcept override;

    // override IPluginV2's virtual functions
    AsciiChar const * getPluginType () const noexcept override;
    
    AsciiChar const * getPluginVersion () const noexcept override;
    
    int32_t getNbOutputs () const noexcept override;
    
    int32_t initialize () noexcept override;
    
    void terminate () noexcept override;
    
    size_t getSerializationSize () const noexcept override;
    
    void serialize (void *buffer) const noexcept override;
    
    void destroy () noexcept override;
    
    void setPluginNamespace (AsciiChar const *pluginNamespace) noexcept override;
    
    AsciiChar const * getPluginNamespace () const noexcept override;

private:
    const std::string mLayerName;   // Name of the custom layer onject
};

class QueryBallPointPluginCreator : public IPluginCreator
{
public:
    QueryBallPointPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

class GroupPointPluginCreator : public IPluginCreator
{
public:
    GroupPointPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

class ClipPlugin : public IPluginV2
{
public:
    ClipPlugin(const std::string name, float clipMin, float clipMax);

    ClipPlugin(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make ClipPlugin without arguments, so we delete default constructor.
    ClipPlugin() = delete;

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int) const noexcept override
    {
        return 0;
    };

    int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type,
        PluginFormat format, int maxBatchSize) noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::IPluginV2* clone() const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    const std::string mLayerName;
    float mClipMin, mClipMax;
    size_t mInputVolume;
    std::string mNamespace;
};

class ClipPluginCreator : public IPluginCreator
{
public:
    ClipPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

#endif
