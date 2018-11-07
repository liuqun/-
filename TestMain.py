# -*- encoding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import contextlib


class MyClassifierModel:
    sess = None
    x = None
    y = None

    def __init__(self, sess, x, y):
        self.x = x
        self.y = y
        self.sess = sess

    def predict(self, data):
        """使用当前会话执行一次预测

        :param data: 待预测数据, 矩阵形状必须是 n*1*80
        :type data: numpy.ndarray
        :return: 预测的负荷索引
        :rtype : numpy.ndarray
        """
        return self.sess.run(self.y, feed_dict={self.x: data})


class ModelFileLoader:
    def __init__(self):
        pass

    @staticmethod
    @contextlib.contextmanager
    def open(ckpt_dir):
        if not os.path.isdir(ckpt_dir):
            raise ModelLoadingError('Invalid path "{}"'.format(ckpt_dir))
        latest_checkpoint_str = tf.train.latest_checkpoint(ckpt_dir)
        if not latest_checkpoint_str:
            raise ModelLoadingError('Checkpoint file not found in dir "{}"'.format(ckpt_dir))
        meta_graph_filename = latest_checkpoint_str + '.meta'
        graph = tf.Graph()
        try:
            with graph.as_default():
                saver = tf.train.import_meta_graph(meta_graph_filename)
        except IOError:
            raise ModelLoadingError('Meta graph file "{}" is missing'.format(meta_graph_filename))
        x = graph.get_tensor_by_name("x:0")
        logits = graph.get_tensor_by_name("logits_eval:0")
        y = tf.argmax(logits, 1)
        sess = tf.Session(graph=graph)
        saver.restore(sess, latest_checkpoint_str)
        pre_trained_model = MyClassifierModel(sess, x, y)
        yield pre_trained_model
        sess.close()


class ModelLoadingError(Exception):
    pass


def print_output_index(output_index):
    if not output_index.any():
        print('预测失败')
    for i in range(len(output_index)):
        print("第", i + 1, "个电器预测:" + appliance[int(output_index[i])])


if __name__ == '__main__':
    cwd = os.getcwd()

    # 模型相关文件存放路径(需修改)
    path_for_check = r"./模型/"

    # 测试数据(需修改)
    path_for_test_data = "./测试样本.txt"

    # 负荷名称(不需修改)
    appliance = {0: 'aux电饭煲-开始', 1: 'ipad air2-充电', 2: '吹风机-2档热1档风', 3: '戴尔E6440',
                 4: '挂烫机-1档', 5: '华为P9Plus充电', 6: '九阳电饭煲-蒸煮', 7: '空调-吹风', 8: '空调-制冷',
                 9: '联想扬天(台式机 显示器)', 10: '水壶', 11: '无电器运行', 12: '吸尘器'}

    # 数据输入参数(不需修改)
    seq_len = 80
    input_dim = 1

    # 测试数据加载与转换(不需修改)
    test_data = np.loadtxt(path_for_test_data)
    rsp_test_data = np.reshape(test_data, (-1, input_dim, seq_len))  # 数据维度处理

    # 模型预测
    try:
        with ModelFileLoader.open(path_for_check) as model:
            output_index = model.predict(rsp_test_data)
        print_output_index(output_index)
    except ModelLoadingError as e:
        print('Debug: Failed:', e)
        exit(127)
