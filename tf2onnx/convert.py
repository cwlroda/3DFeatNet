import argparse
import os
import sys
from distutils.version import LooseVersion

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"    # TF only reports on error.

import tensorflow as tf

from tf2onnx.tfonnx import process_tf_graph
from tf2onnx import constants, logging, utils, optimizer
from tf2onnx import tf_loader
from tf2onnx.graph import ExternalTensorStorage
from tf2onnx.tf_utils import compress_graph_def

# Load in custom ops
tf.load_op_library('./3DFeatNet/tf_ops/grouping/tf_grouping_so.so')
tf.load_op_library('./3DFeatNet/tf_ops/sampling/tf_sampling_so.so')
from models.pointnet_common import sample_points, sample_and_group, sample_and_group_all, query_and_group_points

_HELP_TEXT = "Consult the actual tf2onnx documentation instead of this stripped-down hack."

# Import the custom model from specified path.
def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser(description="Convert tensorflow graphs to ONNX.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_HELP_TEXT)
    parser.add_argument("--saved-model", help="input from saved model")
    parser.add_argument("--tag", help="tag to use for saved_model")
    parser.add_argument("--signature_def", help="signature_def from saved_model to use")
    parser.add_argument("--concrete_function", type=int, default=None,
                        help="For TF2.x saved_model, index of func signature in __call__ (--signature_def is ignored)")
    parser.add_argument("--output", help="output model file")
    parser.add_argument("--inputs", help="model input_names (optional for saved_model, keras, and tflite)")
    parser.add_argument("--outputs", help="model output_names (optional for saved_model, keras, and tflite)")
    parser.add_argument("--ignore_default", help="comma-separated list of names of PlaceholderWithDefault "
                                                 "ops to change into Placeholder ops")
    parser.add_argument("--use_default", help="comma-separated list of names of PlaceholderWithDefault ops to "
                                              "change into Identity ops using their default value")
    parser.add_argument("--rename-inputs", help="input names to use in final model (optional)")
    parser.add_argument("--rename-outputs", help="output names to use in final model (optional)")
    parser.add_argument("--use-graph-names", help="(saved model only) skip renaming io using signature names",
                        action="store_true")
    parser.add_argument("--opset", type=int, default=None, help="opset version to use for onnx domain")
    parser.add_argument("--dequantize", help="Remove quantization from model. Only supported for tflite currently.",
                        action="store_true")
    parser.add_argument("--custom-ops", help="comma-separated map of custom ops to domains in format OpName:domain")
    parser.add_argument("--extra_opset", default=None,
                        help="extra opset with format like domain:version, e.g. com.microsoft:1")
    parser.add_argument("--load_op_libraries",
                        help="comma-separated list of tf op library paths to register before loading model")
    parser.add_argument("--target", default=",".join(constants.DEFAULT_TARGET), choices=constants.POSSIBLE_TARGETS,
                        help="target platform")
    parser.add_argument("--continue_on_error", help="continue_on_error", action="store_true")
    parser.add_argument("--verbose", "-v", help="verbose output, option is additive", action="count")
    parser.add_argument("--debug", help="debug mode", action="store_true")
    parser.add_argument("--output_frozen_graph", help="output frozen tf graph to file")
    
    args = parser.parse_args()
    return args

def make_default_custom_op_handler(domain):
    '''
        Returns a function that acts as a handler for a given custom op.
        This is the key part we are trying to solve
         - to register an appropriate custom op for QueryBallPoint and GroupPoint.
    '''
    def default_custom_op_handler(ctx, node, name, args):
        node.domain = domain
        return node
    return default_custom_op_handler

_TENSORFLOW_DOMAIN = "ai.onnx.converters.tensorflow"
def print_handler(ctx, node, name, args):
    # replace tf.Print() with Identity
    #   T output = Print(T input, data, @list(type) U, @string message, @int first_n, @int summarize)
    # becomes:
    #   T output = Identity(T Input)
    node.type = "Identity"
    node.domain = _TENSORFLOW_DOMAIN
    del node.input[1:]
    return node

def queryBallPoint_handler(ctx, node, name, args):
    pass

def groupPoint_handler(ctx, node, name, args):
    pass

def _convert_common(frozen_graph, name="unknown", large_model=False, output_path=None,
                    output_frozen_graph=None, **kwargs):
    """Common processing for conversion."""

    model_proto = None
    external_tensor_storage = None
    const_node_values = None

    with tf.Graph().as_default() as tf_graph:
        if large_model:
            const_node_values = compress_graph_def(frozen_graph)
            external_tensor_storage = ExternalTensorStorage()
        if output_frozen_graph:
            utils.save_protobuf(output_frozen_graph, frozen_graph)
        if not kwargs.get("tflite_path"):
            tf.import_graph_def(frozen_graph, name='')
        g = process_tf_graph(tf_graph, const_node_values=const_node_values, **kwargs)
        onnx_graph = optimizer.optimize_graph(g, catch_errors=not large_model)
        model_proto = onnx_graph.make_model("converted from {}".format(name),
                                            external_tensor_storage=external_tensor_storage)
    if output_path:
        if large_model:
            utils.save_onnx_zip(output_path, model_proto, external_tensor_storage)
        else:
            utils.save_protobuf(output_path, model_proto)

    return model_proto, external_tensor_storage

def main():
    args = get_args()
    logging.basicConfig(level=logging.get_verbosity_level(args.verbose))
    if args.debug:
        utils.set_debug_mode(True)

    logger = logging.getLogger(constants.TF2ONNX_PACKAGE_NAME)

    extra_opset = args.extra_opset or []
    custom_ops = {}
    initialized_tables = None
    tensors_to_rename = {}
    if args.custom_ops:
        using_tf_opset = False
        for op in args.custom_ops.split(","):
            logger.info("Found custom op {}".format(op))
            if ":" in op:
                op, domain = op.split(":")
            else:
                # default custom ops for tensorflow-onnx are in the "tf" namespace
                using_tf_opset = True
                domain = constants.TENSORFLOW_OPSET.domain
            custom_ops[op] = (make_default_custom_op_handler(domain), [])

        if using_tf_opset:
            extra_opset.append(constants.TENSORFLOW_OPSET)

    # If any op is in the contrib ops domain, then attempt to import tensorflow_text.
    if any(opset.domain == constants.CONTRIB_OPS_DOMAIN for opset in extra_opset):
        try:
            import tensorflow_text   # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError:
            logger.warning("tensorflow_text not installed. Model will fail to load if tensorflow_text ops are used.")

    # get the frozen tensorflow model from saved_model.
    graph_def = None
    inputs = None
    outputs = None
    model_path = None

    if not utils.is_cpp_protobuf():
        logger.warning("***IMPORTANT*** Installed protobuf is not cpp accelerated. Conversion will be extremely slow. "
                       "See https://github.com/onnx/tensorflow-onnx/issues/1557")

    # Loads a SavedModel by default - tuned to our TensorFlow2 SavedModel implementation.
    graph_def, inputs, outputs, initialized_tables, tensors_to_rename = tf_loader.from_saved_model(
            args.saved_model, args.inputs, args.outputs, args.tag, args.signature_def, args.concrete_function,
            args.large_model, return_initialized_tables=True, return_tensors_to_rename=True,
            use_graph_names=args.use_graph_names)
    
    model_path = args.saved_model

    if args.verbose:
        logger.info("inputs: %s", inputs)
        logger.info("outputs: %s", outputs)

    if args.rename_inputs:
        tensors_to_rename.update(zip(inputs, args.rename_inputs))
    if args.rename_outputs:
        tensors_to_rename.update(zip(outputs, args.rename_outputs))

    with tf.device("/cpu:0"):
        model_proto, _ = _convert_common(
            graph_def,
            name=model_path,
            continue_on_error=args.continue_on_error,
            target=args.target,
            opset=args.opset,
            custom_op_handlers=custom_ops,
            extra_opset=extra_opset,
            shape_override=args.shape_override,
            input_names=inputs,
            output_names=outputs,
            inputs_as_nchw=args.inputs_as_nchw,
            large_model=args.large_model,
            tensors_to_rename=tensors_to_rename,
            ignore_default=args.ignore_default,
            use_default=args.use_default,
            dequantize=args.dequantize,
            initialized_tables=initialized_tables,
            output_frozen_graph=args.output_frozen_graph,
            output_path=args.output)

    # write onnx graph
    logger.info("")
    logger.info("Successfully converted TensorFlow model %s to ONNX", model_path)

    logger.info("Model inputs: %s", [n.name for n in model_proto.graph.input])
    logger.info("Model outputs: %s", [n.name for n in model_proto.graph.output])
    if args.output:
        if args.large_model:
            logger.info("Zipped ONNX model is saved at %s. Unzip before opening in onnxruntime.", args.output)
        else:
            logger.info("ONNX model is saved at %s", args.output)
    else:
        logger.info("To export ONNX model to file, please run with `--output` option")


""" 
#Register TF custom ops.

_TENSORFLOW_DOMAIN = "ai.onnx.converters.tensorflow"


def print_handler(ctx, node, name, args):
    # replace tf.Print() with Identity
    #   T output = Print(T input, data, @list(type) U, @string message, @int first_n, @int summarize)
    # becomes:
    #   T output = Identity(T Input)
    node.type = "Identity"
    node.domain = _TENSORFLOW_DOMAIN
    del node.input[1:]
    return node

def _handler():
    pass


with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [2, 3], name="input")
    x_ = tf.add(x, x)
    x_ = tf.Print(x_, [x_], "hello")
    _ = tf.identity(x_, name="output")
    onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph,
                                                 custom_op_handlers={"Print": (print_handler, [])},
                                                 extra_opset=[helper.make_opsetid(_TENSORFLOW_DOMAIN, 1)],
                                                 input_names=["input:0"],
                                                 output_names=["output:0"])
    model_proto = onnx_graph.make_model("test")
    with open("/tmp/model.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())


tf2onnx.convert.main()
"""

if __name__ == "__main__":
    main()