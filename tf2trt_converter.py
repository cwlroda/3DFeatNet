import argparse
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

parser = argparse.ArgumentParser(description="Converts model to TensorRT")
parser.add_argument(
    "--format", type=str, default="ckpt2trt", help="Conversion from .ckpt or .pb"
)
parser.add_argument(
    "--meta_path",
    type=str,
    default="./ckpt/secondstage/ckpt/checkpoint.ckpt-210000.meta",
    help="Path to .meta file",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    default="./ckpt/secondstage/ckpt/checkpoint.ckpt-210000",
    help="Path to .ckpt file",
)
parser.add_argument(
    "--pb_path",
    type=str,
    default="./export_dir/0/saved_model.pb",
    help="Path to .pb file",
)
args = parser.parse_args()

outputs = ["save_1/SaveV2"]


def ckpt2trt():
    # First create a `Saver` object (for saving and rebuilding a
    # model) and import your `MetaGraphDef` protocol buffer into it:
    saver = tf.train.import_meta_graph(args.meta_path)

    # Then restore your training data from checkpoint files:
    saver.restore(sess, args.ckpt_path)

    # Finally, freeze the graph:
    frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess, tf.get_default_graph().as_graph_def(), output_node_names=outputs,
    )

    tf.train.write_graph(
        frozen_graph, "./export_dir/0/", "saved_model.pb", as_text=False
    )

    converter = trt.TrtGraphConverter(
        input_graph_def=frozen_graph, nodes_blacklist=outputs
    )
    trt_graph = converter.convert()

    return trt_graph


def pb2trt():
    # First deserialize your frozen graph:
    with tf.gfile.GFile(args.pb_path, "rb") as f:
        frozen_graph = tf.GraphDef()
        frozen_graph.ParseFromString(f.read())

    # Now you can create a TensorRT inference graph from your
    # frozen graph:
    converter = trt.TrtGraphConverter(
        input_graph_def=frozen_graph, nodes_blacklist=outputs
    )
    trt_graph = converter.convert()

    return trt_graph


if __name__ == "__main__":
    with tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
    ) as sess:
        if args.format == "ckpt2trt":
            from tf_ops.grouping.tf_grouping import (
                query_ball_point,
                group_point,
                knn_point,
            )
            from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point

            trt_graph = ckpt2trt()

        else:
            trt_graph = pb2trt()

        # Import the TensorRT graph into a new graph and run:
        # g = tf.get_default_graph()
        # TODO: get input tensor name
        # inputs = g.get_tensor_by_name("")
        # TODO: map inputs
        # output_node = tf.import_graph_def(trt_graph, input_map={}, return_elements=outputs)
        # print("Output node: ")
        # print(output_node)
        # TODO: add input node to feed_dict
        # sess.run(output_node, feed_dict={})
