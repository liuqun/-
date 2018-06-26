# -*- encoding: utf-8 -*-
import numpy as np
import os.path
import tensorflow as tf


class AppliancePredictor():
    def __init__(self):
        self.checkpoint_str = None
        self.model_saver = None

    def load_model(self, chk_path):
        if not os.path.isdir(chk_path):
            raise ValueError('Error: Invalid value for chk_path: "%s" is not a directory!' % chk_path)
        try:
            self.checkpoint_str = tf.train.latest_checkpoint(chk_path)
        except IOError as e:
            raise RuntimeError('Error: "%s": %s' % (chk_path, e))
        path_for_meta = self.checkpoint_str + '.meta'
        try:
            self.model_saver = tf.train.import_meta_graph(path_for_meta)
        except IOError as e:
            raise RuntimeError('Error: Meta file "%s" is missing: %s' % (path_for_meta, e))

    def predict(self, data) -> np.ndarray:
        """预测输入数据对应的家电类型编号
        """
        if not self.model_saver or not self.checkpoint_str:
            raise RuntimeError('Error: Model is not loaded yet!')
        with tf.Session() as sess:
            self.model_saver.restore(sess, self.checkpoint_str)
            graph = tf.get_default_graph()
            input_name = graph.get_tensor_by_name("x:0")
            feed_dict = {input_name: data}
            logits = graph.get_tensor_by_name("logits_eval:0")
            argmax = tf.argmax(logits, 1)
            result = sess.run(argmax, feed_dict)
        return result
