from resnet_v1 import resnet_v1, resnet_arg_scope, resnet_v1_50
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def main():
    ckpt_path = './resnet_v1_50.ckpt'
    X = tf.placeholder(tf.float32, shape=[None, 96, 96, 3], name='input')

    with slim.arg_scope(resnet_arg_scope()):
        logits, end_points = resnet_v1_50(X, num_classes=1000, is_training=False)

    final_layer_to_load = end_points['resnet_v1_50/block4']

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)
        frozen_graph_def = convert_variables_to_constants(
            sess, sess.graph_def,
            output_node_names=[final_layer_to_load.name.split(':')[0]])

    frozen_graph = tf.Graph()
    with frozen_graph.as_default():
        tf.import_graph_def(frozen_graph_def, name='')

    sess = tf.Session(graph=frozen_graph)

    res = sess.run(final_layer_to_load.name, {'input:0': np.ones(shape=[12, 96, 96, 3])})
    print("out shape: {}".format(res.shape))
    
if __name__ == "__main__":
    main()
