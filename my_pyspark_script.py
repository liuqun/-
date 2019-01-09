#!/usr/bin/env python2
# -*- endcoding: utf-8 -*-
r"""
 To run this on your local machine, you need to first run a Netcat server
    `$ nc -lk 9999`
 and then run the example
    `$ bin/spark-submit path/to/my_pyspark_script.py localhost 9999`
"""
from __future__ import print_function
from __future__ import division
import contextlib

import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

import numpy as np
import tensorflow as tf
import os.path
from datetime import datetime
import socket
from socket import AF_INET, SOCK_DGRAM
from operator import add


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
    model_instance = None

    def __init__(self):
        pass

    @staticmethod
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
        return pre_trained_model

    @staticmethod
    def get_instance():
        if not ModelFileLoader.model_instance is None:
            return ModelFileLoader.model_instance
        ModelFileLoader.model_instance = ModelFileLoader.open('/models')
        return ModelFileLoader.model_instance

class ModelLoadingError(Exception):
    pass


def reshape_ndarray(a):
    print('== Debug: timestamp =', datetime.now())

    print('== Debug: input size =', a.size)
    INPUT_DIM = 1
    SEQ_LEN = 80
    if a.size % 80 != 0:
        a = np.resize(a, int(a.size // SEQ_LEN) * 80)
    return a.reshape((int(a.size // SEQ_LEN), INPUT_DIM, SEQ_LEN))


# 负荷名称(不需修改)
APPLIANCE_NAME_TABLE = [
    '【aux电饭煲】',
    '【iPad air2-充电】',
    '【吹风机-2档热1档风】',
    '【戴尔E6440台式电脑】',
    '【挂烫机-1档】',
    '【华为P9Plus充电】',
    '【九阳电饭煲-蒸煮】',
    '【空调-吹风】',
    '【空调-制冷】',
    '【联想扬天(台式机+显示器)】',
    '【水壶】',
    '无电器运行',
    '【吸尘器】',
]
JIA_DIAN_ZHONG_LEI_SHU = len(APPLIANCE_NAME_TABLE)


def do_model_classify(reshaped_data):
    model = ModelFileLoader.get_instance()
    return model.predict(reshaped_data)


def do_result_counting(uncounted_result_list):
    global JIA_DIAN_ZHONG_LEI_SHU
    cnt = np.zeros(JIA_DIAN_ZHONG_LEI_SHU, np.int32)
    for x in uncounted_result_list:
        cnt[x] += 1
    return cnt


global APPLIANCE_NAME_TABLE_GBK
APPLIANCE_NAME_TABLE_GBK = [utf8.decode('utf-8').encode('gbk') for utf8 in APPLIANCE_NAME_TABLE]

def gbk_msg_from_table(table):
    global APPLIANCE_NAME_TABLE_GBK
    tokens = []
    for key, cnt in table:
        if cnt <= 0:
            continue
        tokens.append(b'%s*%d' % (APPLIANCE_NAME_TABLE_GBK[key], cnt))
    msg = ', '.join(tokens)
    return msg

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:\n spark-submit {script} 192.168.1.158 9999".format(script=sys.argv[0]), file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="TestModelClassify")
    BATCH_DURATION_SECONDS = 10
    ssc = StreamingContext(sc, BATCH_DURATION_SECONDS)

    text_stream_in = ssc.socketTextStream(sys.argv[1], int(sys.argv[2]))
    collection = text_stream_in \
            .filter(lambda line: len(line.strip()) > 0) \
            .map(lambda line: np.fromstring(line, dtype=np.float, sep=" ")) \
            .map(reshape_ndarray) \
            .filter(lambda x: int(x.size % 80) == 0) \
            .map(do_model_classify) \
            .map(do_result_counting
            ).reduce(add
            ).map(lambda counters:
                    sorted(zip(range(12), counters), key=lambda item: item[1], reverse=True))


    udpremotehostport = ('192.168.1.158', 8888)
    udpremotetimeout = 10
    def my_print(time, rdd):
        def send_msg(iter):
            udpout = socket.socket(AF_INET, SOCK_DGRAM)
            udpout.sendto(b''.join(('=== ', time.strftime('%Y-%m-%d %H:%M:%S'), ' ===\n\r')), udpremotehostport)
            for sorted_table in iter:
                gbkmsg = b''.join((gbk_msg_from_table(sorted_table), '\n\r'))
                udpout.sendto(gbkmsg, udpremotehostport)
            udpout.sendto(b'\n\r', udpremotehostport)
            udpout.close()
        if rdd.count() > 0:
            rdd.foreachPartition(send_msg)
    collection.foreachRDD(my_print)

    ssc.start()
    ssc.awaitTermination()
