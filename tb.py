import argparse
import tensorflow as tf
from tf_ops.grouping.tf_grouping import (
    query_ball_point,
    group_point,
    knn_point,
)
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point

parser = argparse.ArgumentParser(description="Converts model to TensorRT")
parser.add_argument(
    "--meta_path",
    type=str,
    default="./ckpt/secondstage/ckpt/checkpoint.ckpt-210000.meta",
    help="Path to .meta file",
)
args = parser.parse_args()

tf.train.import_meta_graph(args.meta_path)
for n in tf.get_default_graph().as_graph_def().node:
    print(n)

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./__tb/", sess.graph)
    writer.close()
