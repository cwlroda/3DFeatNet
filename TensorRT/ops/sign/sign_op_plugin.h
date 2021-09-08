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

 * Template by NVIDIA, developed by Tianyi.
 * Declares functions for the Sign Function.
 */

#ifndef SIGN_OP_PLUGIN_H
#define SIGN_OP_PLUGIN_H

#include "NvInferPlugin.h"
#include "sign_op_kernel.h"
#include <string>
#include <vector>

using namespace nvinfer1;

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class SignOpPlugin : public IPluginV2DynamicExt
{
public:
    // Constructor for concrete parameters
    SignOpPlugin(const std::string name);

    // Constructor for serialised data
    SignOpPlugin(const std::string name, const void* data, size_t length);

    // doesn't make sense to call the constructor w/o args, so delete default constructor
    SignOpPlugin() = delete;

    // ~ override IPluginV2DynamicExt's virtual functions

    IPluginV2DynamicExt * 	clone () const noexcept override;

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

    // ~ override IPluginV2Ext's virtual functions

    nvinfer1::DataType getOutputDataType (int32_t index, nvinfer1::DataType const *inputTypes, 
                    int32_t nbInputs) const noexcept override;

    void attachToContext (cudnnContext *, cublasContext *, IGpuAllocator *) noexcept override;
    
    void detachFromContext () noexcept override;

    // ~ override IPluginV2's virtual functions

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
    const std::string mLayerName;   // Name of the custom layer 
    std::string mNamespace;         // Name of custom namespace
};

class SignOpPluginCreator : public IPluginCreator
{
public:
    SignOpPluginCreator();

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