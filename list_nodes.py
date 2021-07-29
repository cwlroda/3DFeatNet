import tensorflow as tf
from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point  # Import custom ops

TF_VERSION = tf.__version__
if TF_VERSION[0]=="2":
    print("Using Tensorflow 2")

    sess = tf.compat.v1.Session()
    tf.compat.v1.disable_eager_execution() # To ensure meta graph can be generated
    tf.compat.v1.train.import_meta_graph("../ckpt/checkpoint.ckpt-210000.meta") # Change this wrt the file structure of your directory
    writer = tf.summary.create_file_writer("__tb")
    with writer.as_default():
        graph_def = tf.compat.v1.get_default_graph().as_graph_def()
        node_list = [n.name for n in graph_def.node]
        outputs = set(node_list)

        for index, output in enumerate(outputs):
            print("Node Name: ", output)

else:
    print("Using Tensorflow 1")
    from tensorflow.summary import FileWriter

    sess = tf.Session()
    tf.train.import_meta_graph("./ckpt/secondstage/ckpt/checkpoint.ckpt-210000.meta")
    FileWriter("__tb", sess.graph)
    graph_def = tf.get_default_graph().as_graph_def()
    node_list = [n.name for n in graph_def.node]
    outputs = set(node_list)

    for index, output in enumerate(outputs):
        print("Node Name: ", output)
