import tensorflow as tf
from tensorflow.summary import FileWriter
from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point

sess = tf.Session()
tf.train.import_meta_graph("./ckpt/secondstage/ckpt/checkpoint.ckpt-210000.meta")
FileWriter("__tb", sess.graph)
graph_def = tf.get_default_graph().as_graph_def()
node_list = [n.name for n in graph_def.node]
outputs = set(node_list)

for index, output in enumerate(outputs):
    print("Node Name: ", output)
