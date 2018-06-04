# -*- encoding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os


def model_predict(chk_path, data):
    '''
    模型预测
    :param chk_path: checkpoint文件所在路径
    :param data: 待预测数据(-1, 数据维度, 数据长度)
    :return: 预测的负荷索引
    '''
    if not os.path.isdir(chk_path):
        return False

    with tf.Session() as sess:
        try:
            latest_checkpoint_str = tf.train.latest_checkpoint(chk_path)
            path_for_meta = latest_checkpoint_str + '.meta'
            saver = tf.train.import_meta_graph(path_for_meta)
        except IOError:
            print('meta file missing')
            return False

        saver.restore(sess, tf.train.latest_checkpoint(chk_path))
        graph = tf.get_default_graph()

        input_name = graph.get_tensor_by_name("x:0")
        logits = graph.get_tensor_by_name("logits_eval:0")

        feed_dict = {input_name: data}
        classification_result = sess.run(logits, feed_dict)

        return tf.argmax(classification_result, 1).eval()


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
    rsp_test_data = np.reshape(test_data, (int(len(test_data) / input_dim), input_dim, seq_len))  # 数据维度处理

    # 模型预测
    output_index = model_predict(path_for_check, rsp_test_data)

    # 结果输出
    if output_index.any():
        for i in range(len(output_index)):
            print("第", i + 1, "个电器预测:" + appliance[int(output_index[i])])
    else:
        print('预测失败')
