#!/bin/bash

ONNX_OPSET=13
MODEL_SAVEPATH="./inference_savedmodel"
ONNX_SAVEPATH="./onnx_models"

TF_CPP_MIN_LOG_LEVEL='2'  # Squelch INFO and DEBUG messages

# Run inference on dummy data
python inference.py             \
    --data_dir='./example_data' \
    --output_dir='./example_data/results'   \
    --checkpoint='./ckpt/secondstage/ckpt' \
    --randomize_points \
    --model_savepath=${MODEL_SAVEPATH}


set -e # Abort if any of these fail

echo -e "\n\n\n##############################################"
echo "#### Converting Detect and Describe model ####"
echo -e "##############################################\n\n\n"

python -m tf2onnx.convert \
--saved-model ${MODEL_SAVEPATH}/det_desc \
--output ${ONNX_SAVEPATH}/model_det_desc.onnx \
--load_op_libraries ./tf_ops/grouping/tf_grouping_so.so,./tf_ops/sampling/tf_sampling_so.so \
--rename-inputs in_keypoints,in_pointcloud 
--rename-outputs out_keypoints,out_features,out_attention \
--custom-ops QueryBallPoint,GroupPoint \
--opset ${ONNX_OPSET} --target tensorrt

echo -e "\n\n\n##############################################"
echo "#### Converting Detector model ####"
echo -e "##############################################\n\n\n"

python -m tf2onnx.convert \
--saved-model ${MODEL_SAVEPATH}/det_only \
--output ${ONNX_SAVEPATH}/model_det_only.onnx \
--load_op_libraries ./tf_ops/grouping/tf_grouping_so.so,./tf_ops/sampling/tf_sampling_so.so \
--rename-inputs in_pointcloud
--rename-outputs out_keypoints,out_attention,out_orientation \
--custom-ops QueryBallPoint,GroupPoint \
--opset ${ONNX_OPSET} --target tensorrt

echo -e "\n\n\n##############################################"
echo "#### Converting Descriptor model ####"
echo -e "##############################################\n\n\n"

python -m tf2onnx.convert \
--saved-model ${MODEL_SAVEPATH}/desc_only \
--output ${ONNX_SAVEPATH}/model_desc_only.onnx \
--load_op_libraries ./tf_ops/grouping/tf_grouping_so.so,./tf_ops/sampling/tf_sampling_so.so \
--rename-inputs in_keypoints,in_pointcloud,in_orientation \
--rename-outputs out_keypoints,out_features \
--custom-ops QueryBallPoint,GroupPoint \
--opset ${ONNX_OPSET} --target tensorrt