# ONNX Runtime example in python
import onnxruntime as rt
import onnx
import numpy as np
import os
GROUPING_LIB = os.path.join(os.getcwd(), "tf_ops", "grouping", "tf_grouping_so.so")
SAMPLING_LIB = os.path.join(os.getcwd(), "tf_ops", "sampling", "tf_sampling_so.so")

# Print some info about the runtime
print("Runtime is on:", rt.get_device())
# print("ONNX is on:", )

# model_data = rt.ModelMetadata()
# print("""
#     Model Graph Name: {}
#     Model description: {}
#     Model domain: {}
#     Model Graph Description: {}
#     """.format(model_data.graph_name, model_data.description,
#         model_data.domain, model_data.graph_description
#     )
#     )

# Need to register the custom ops here somehow.
# This fails here because it is not registered in the ONNX format.
opts = rt.SessionOptions()
opts.register_custom_ops_library(GROUPING_LIB)
opts.register_custom_ops_library(SAMPLING_LIB)

sess = rt.InferenceSession("onnx_models/model_infer.onnx", sess_options=opts)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
print("Model inputs: {}\nModel outputs: {}".format(input_name, label_name))
# pred = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]