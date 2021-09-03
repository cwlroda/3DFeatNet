#!/bin/bash

TF_CPP_MIN_LOG_LEVEL='2'  # Squelch INFO and DEBUG messages
python inference.py             \
    --data_dir='./example_data' \
    --output_dir='./example_data/results'   \
    --checkpoint='./ckpt/secondstage/ckpt' \
    --randomize_points \
    --model_savepath="./inference_savedmodel"

ONNX_OPSET=13

echo -e "\n\n\n##############################################"
echo "#### Converting Detect and Describe model ####"
echo -e "##############################################\n\n\n"

python -m tf2onnx.convert \
--saved-model ./inference_savedmodel/det_desc --output onnx_models/model_det_desc.onnx \
--load_op_libraries ./tf_ops/grouping/tf_grouping_so.so,./tf_ops/sampling/tf_sampling_so.so \
--rename-inputs pointcloud --rename-outputs keypoints,features,attention \
--custom-ops QueryBallPoint,GroupPoint --opset ${ONNX_OPSET} --target tensorrt

echo -e "\n\n\n##############################################"
echo "####### Converting Describe-only model #######"
echo -e "##############################################\n\n\n"

python -m tf2onnx.convert \
--saved-model ./inference_savedmodel/desc_only --output onnx_models/model_desc_only.onnx \
--load_op_libraries ./tf_ops/grouping/tf_grouping_so.so,./tf_ops/sampling/tf_sampling_so.so \
--rename-outputs keypoints,features \
--custom-ops QueryBallPoint,GroupPoint --opset ${ONNX_OPSET} --target tensorrt