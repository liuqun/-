#!/usr/bin/env python
# -*- endcoding: utf-8 -*-
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

r"""
 Counts words in UTF8 encoded, '\n' delimited text received from the network every 20 seconds.
 Usage: network_wordcount.py <hostname> <port>
   <hostname> and <port> describe the TCP server that Spark Streaming would connect to receive data.

 To run this on your local machine, you need to first run a Netcat server
    `$ nc -lk 9999`
 and then run the example
    `$ bin/spark-submit examples/src/main/python/streaming/network_wordcount.py localhost 9999`
"""
from __future__ import print_function
from __future__ import division

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

class MyVariables:
    x = None
    logits = None
    y = None


def init_tf_session():
    PATH_FOR_META = '/models/model.ckpt-50.meta'
    path_for_model_dir = os.path.dirname(PATH_FOR_META)

    #graph1 = tf.Graph()
    #with graph1.as_default():
    #    saver = tf.train.import_meta_graph(PATH_FOR_META)
    saver = tf.train.import_meta_graph(PATH_FOR_META)
    graph = tf.get_default_graph()
    local_vars = MyVariables()
    local_vars.x = graph.get_tensor_by_name("x:0")
    local_vars.logits = graph.get_tensor_by_name("logits_eval:0")
    local_vars.y = tf.argmax(local_vars.logits, 1)

    #session1 = tf.Session(graph=graph1)
    #saver.restore(session1, tf.train.latest_checkpoint(path_for_model_dir))
    session2 = tf.Session()
    saver.restore(session2, tf.train.latest_checkpoint(path_for_model_dir))
    return session2, local_vars


def do_tf_warming_up():
    test_str = """6.540000000000000036e+00 3.271000000000000085e+01 6.540999999999999659e+01 9.893000000000000682e+01 1.275499999999999972e+02 1.618899999999999864e+02 1.815099999999999909e+02 2.003199999999999932e+02 2.207599999999999909e+02 2.379300000000000068e+02 2.681800000000000068e+02 2.820799999999999841e+02 3.008899999999999864e+02 3.139700000000000273e+02 3.049700000000000273e+02 3.033399999999999750e+02 3.025199999999999818e+02 3.041600000000000250e+02 3.025199999999999818e+02 3.016999999999999886e+02 3.041600000000000250e+02 3.016999999999999886e+02 3.008899999999999864e+02 3.000699999999999932e+02 3.000699999999999932e+02 2.984300000000000068e+02 3.008899999999999864e+02 2.910699999999999932e+02 2.714499999999999886e+02 2.485600000000000023e+02 2.158499999999999943e+02 1.888700000000000045e+02 1.610699999999999932e+02 1.250999999999999943e+02 8.993999999999999773e+01 5.560000000000000142e+01 2.698000000000000043e+01 0.000000000000000000e+00 -3.025000000000000000e+01 -5.886999999999999744e+01 -8.829999999999999716e+01 -1.250999999999999943e+02 -1.504399999999999977e+02 -1.766100000000000136e+02 -2.019499999999999886e+02 -2.142199999999999989e+02 -2.330200000000000102e+02 -2.526500000000000057e+02 -2.747200000000000273e+02 -2.918899999999999864e+02 -3.074300000000000068e+02 -3.098799999999999955e+02 -3.098799999999999955e+02 -3.082500000000000000e+02 -3.074300000000000068e+02 -3.090600000000000023e+02 -3.074300000000000068e+02 -3.098799999999999955e+02 -3.049700000000000273e+02 -3.057900000000000205e+02 -3.049700000000000273e+02 -2.886200000000000045e+02 -3.041600000000000250e+02 -3.066100000000000136e+02 -3.008899999999999864e+02 -2.853500000000000227e+02 -2.616399999999999864e+02 -2.420200000000000102e+02 -2.101299999999999955e+02 -1.790600000000000023e+02 -1.479900000000000091e+02 -1.079300000000000068e+02 -7.521999999999999886e+01 -4.906000000000000227e+01 -8.990000000000000213e+00 8.179999999999999716e+00 3.760999999999999943e+01 7.195000000000000284e+01 1.054699999999999989e+02 1.357299999999999898e+02
"""
    raw_data = np.fromstring(test_str, dtype=np.float64, sep=" ")
    print(raw_data)
    my_reshaped_data = np.reshape(raw_data, (1, 1, 80))
    print(my_reshaped_data)
    sess, my_vars = init_tf_session()
    with sess:
        results = sess.run(my_vars.y, feed_dict={my_vars.x: my_reshaped_data})
    print(results)


#def ch_split(line):
#    return line.split(" ")

# .map(lambda lines: lines.split("\n"))


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
    sess, my_vars = init_tf_session()
    with sess:
        results = sess.run(my_vars.y, feed_dict={my_vars.x: reshaped_data})
        print(results)
    return results

def do_result_counting(uncounted_result_list):
    global JIA_DIAN_ZHONG_LEI_SHU
    cnt = np.zeros(JIA_DIAN_ZHONG_LEI_SHU, np.int32)
    for x in uncounted_result_list:
        cnt[x] += 1
    return cnt


#def output_text(rdd):
#    v = rdd.collect()
#    if len(v) < 1:
#        print('Debug: No data inputed now, timestamp =', datetime.now())
#        return
#    print(v)
#    a = v[0]
#    total_cnt = sum(a)
#    print('Debug: Computed records:', total_cnt)
#    print('Debug: Completed at', datetime.now())

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
        print("Usage: network_wordcount.py <hostname> <port>", file=sys.stderr)
        exit(-1)
    print('skip warming-ups...')
    #do_tf_warming_up()
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
