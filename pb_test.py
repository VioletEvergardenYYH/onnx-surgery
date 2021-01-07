import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='7'
def freeze_graph_test(pb_path):
    with tf.device('/gpu:7'):
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            with open(pb_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")
            config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                # 定义输入的张量名称,对应网络结构的输入张量
                input_tensor = sess.graph.get_tensor_by_name("inputs_placeholder:0")
                input_lens = sess.graph.get_tensor_by_name("input_lens_placeholder:0")
                #out1_tensor = sess.graph.get_tensor_by_name("gec_ged_model_revised_gedloss_1/strided_slice_12:0")
                #out2_tensor = sess.graph.get_tensor_by_name("gec_ged_model_revised_gedloss_1/strided_slice_13:0")
                out_tensor = sess.graph.get_tensor_by_name("intent_labels_probabilities:0")
                # 定义输出的张量名称
                x = np.ones([2,130])
                y = np.ones([2])
                #out1, out2 = sess.run([out1_tensor, out2_tensor], feed_dict = {input_tensor:x})
                #out1 = np.array(out1)
                #out2 = np.array(out2)
                out = sess.run([out_tensor], feed_dict = {input_tensor:x, input_lens:y})
                out = np.array(out)
                x = np.array([178, 1138, 170, 3676, 1, 1])
                x = np.reshape(x, [1, -1, 1, 1])

                out1, out2 = sess.run([out1_tensor, out2_tensor], feed_dict = {input_tensor:x})
                out1 = np.array(out1)
                out2 = np.array(out2)
                print("---------------")
                print(out)
                #print(out1)
                print("---------------")
                #print(out2)
                print("test done")
if __name__ == '__main__':
    pb_path = '/data00/home/yangyuehang/model/correct/capt_gec.pb'
    freeze_graph_test(pb_path)
