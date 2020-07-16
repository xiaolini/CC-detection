from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from form_D_System import Ui_MainWindow

import numpy as np
import csv
from scapy.all import *
import pandas as pd
import os
import dpkt
import datetime

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from predict_input import predict_1152,predict_5,predict_1157
import time
from input_cnn import inputs_1152,inputs_5,inputs_1157

class MyWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(QtWidgets.QMainWindow, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)

    def show_result(self):
        print("Importing dataset...")
        x0, y0 = inputs_1152()  # 导入训练数据集1152
        # x0,y0=inputs_5() #导入训练数据集5
        # x0,y0=inputs_1157()  #导入训练数据集1157
        print("Finished")

        # 分成训练集和测试集 #这里划分数据以1/3的来划分 test_size指测试集占比
        train_X, test_X, train_y, test_y = train_test_split(x0, y0, test_size=1 / 3, random_state=1)
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        [n, x_dim] = train_X.shape
        print(x_dim)

        # 每个批次的大小
        batch_size = 20
        # 计算一共有多少个批次
        n_batch = len(train_X) // batch_size  # 返回一个不大于结果的最大整数

        # 定义两个placeholder
        x = tf.placeholder(tf.float32, [None, x_dim], name='X')  # x列 行不确定
        y = tf.placeholder(tf.float32, [None, 2], name='Y')  # 是ddos=1.或 不是ddos=0

        keep_prob = tf.placeholder(tf.float32,name='keep_prob')

        # 创建一个简单的神经网络  包含两个隐藏层
        W1 = tf.Variable(tf.truncated_normal([x_dim, 500], stddev=0.1))  # 二初始值
        b1 = tf.Variable(tf.zeros([500]) + 0.1)
        L1 = tf.nn.relu(tf.matmul(x, W1) + b1)
        L1_drop = tf.nn.dropout(L1, keep_prob)

        W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))  # 二初始值
        b2 = tf.Variable(tf.zeros([300]) + 0.1)
        L2 = tf.nn.relu(tf.matmul(L1_drop, W2) + b2)
        L2_drop = tf.nn.dropout(L2, keep_prob)

        W3 = tf.Variable(tf.truncated_normal([300, 2], stddev=0.1))  # 二初始值
        b3 = tf.Variable(tf.zeros([2]) + 0.1)
        prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)

        # 二次代价函数
        loss = tf.reduce_mean(tf.square(y - prediction))  # 三交叉熵

        # 使用AdamOptimizer进行优化
        train_step = tf.train.AdamOptimizer(0.00001).minimize(loss)

        # 函数tf.argmax()返回值是是数值最大值的索引位置，如果最大值位置相同，则分类正确，反之则分类错误
        predictions = tf.argmax(prediction, 1)
        actuals = tf.argmax(y, 1)
        # 将上述代码获得到变量设置为元素为0或者1的矩阵，在后面计算的时候只需要按照逻辑与计算即可
        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)

        tp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, ones_like_actuals),
                    tf.equal(predictions, ones_like_predictions)
                ),
                "float"
            )
        )

        tn_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(predictions, zeros_like_predictions)
                ),
                "float"
            )
        )

        fp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(predictions, ones_like_predictions)
                ),
                "float"
            )
        )

        fn_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, ones_like_actuals),
                    tf.equal(predictions, zeros_like_predictions)
                ),
                "float"
            )
        )

        def splitdataset(list, batch):
            return list[batch_size * batch:batch_size * (batch + 1)]

        with tf.Session() as sess:
            start = time.clock()  # 开始计时
            sess.run(tf.global_variables_initializer())
            m_saver = tf.train.Saver()  # 模型保存
            for batch in range(n_batch):
                batch_xs = splitdataset(train_X, batch)
                batch_ys = splitdataset(train_y, batch)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})

            # 假如想要保存hypothesis和cost，以便在保存模型后，重新导入模型时可以使用。
            tp, tn, fp, fn = sess.run([tp_op, tn_op, fp_op, fn_op], feed_dict={x: test_X, y: test_y, keep_prob: 0.7})

            accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))
            recall = float(tp) / (float(tp) + float(fn))
            precision = float(tp) / (float(tp) + float(fp))
            f1_score = (2 * (precision * recall)) / (precision + recall)
            print("accuracy= " + str(accuracy) + ", recall= " + str(recall) + ", precision= " + str(
                precision) + ", f1_score= " + str(f1_score))
            tf.add_to_collection('prediction', prediction)  # 必须有个名字，即第一个参数
            tf.add_to_collection('loss', loss)
            m_saver.save(sess, './model/ddos_detect_model')
            elapsed = (time.clock() - start)  # 结束计时
            print("training time:" + str(elapsed) + "s")  # 印出时间
        self.textEdit_3.setText(str(round(accuracy,6)))
        self.textEdit_4.setText(str(round(precision,6)))
        self.textEdit_5.setText(str(round(recall,6)))
        self.textEdit_6.setText(str(round(f1_score,6)))

    def show_filename(self):
        #获取路径===========================================================
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', 'All files(*.*)')
        global path_openfile_name
        path_openfile_name=openfile_name[0]
        self.textEdit.setText(path_openfile_name)

    def get_basic_info(self):
        pcap_file = path_openfile_name
        print (pcap_file)
        if pcap_file:
            print("1")
            # 获取ip/port等
            src = []
            dst = []
            sport = []
            dport = []
            flags = []
            payload = []
            packages = rdpcap(pcap_file)
            for data in packages:
                if 'TCP' in data:
                    src.append(data['IP'].src)
                    dst.append(data['IP'].dst)
                    sport.append(data['TCP'].sport)
                    dport.append(data['TCP'].dport)
                    flags.append(data['TCP'].flags)
                    payload.append(data.payload)
                else:
                    src.append(data['IP'].src)
                    dst.append(data['IP'].dst)
                    sport.append('-1')
                    dport.append('-1')
                    flags.append('-1')
                    payload.append(data.payload)
            print("2")
            # 获取时间戳
            Timestamp = []
            f = open(pcap_file, 'rb')
            pcap = dpkt.pcap.Reader(f)
            for timestamp, buf in pcap:
                Timestamp.append(str(datetime.datetime.utcfromtimestamp(timestamp)))
            print("3")
            d = {'sip': src, 'dip': dst, 'sport': sport, 'dport': dport, 'flags': flags, 'time-stamp': timestamp,
                 'payload': payload}
            # d={'time-stamp': Timestamp}
            feature = pd.DataFrame(d)
            file_name = "./csv/basic_info.csv"
            feature.to_csv(file_name, index=False)  # 把一个DataFrame写入csv文件
        global path_csvfile
        path_csvfile=file_name

    def creat_table_show(self):
        ###===========读取表格，转换表格，===========================================
        if len(path_csvfile) > 0:
            input_table = pd.read_csv(path_csvfile)
            print(input_table)
            input_table_rows = input_table.shape[0]
            input_table_colunms = input_table.shape[1]
            print(input_table_rows)
            print(input_table_colunms)
            input_table_header = input_table.columns.values.tolist()
            print(input_table_header)

            ###===========读取表格，转换表格，============================================
            ###======================给tablewidget设置行列表头============================

            self.tableWidget.setColumnCount(input_table_colunms)
            self.tableWidget.setRowCount(input_table_rows)
            self.tableWidget.setHorizontalHeaderLabels(input_table_header)

            ###======================给tablewidget设置行列表头============================

            ###================遍历表格每个元素，同时添加到tablewidget中========================
            for i in range(input_table_rows):
                input_table_rows_values = input_table.iloc[[i]]
                # print(input_table_rows_values)
                input_table_rows_values_array = np.array(input_table_rows_values)
                input_table_rows_values_list = input_table_rows_values_array.tolist()[0]
                # print(input_table_rows_values_list)
                for j in range(input_table_colunms):
                    input_table_items_list = input_table_rows_values_list[j]
                    # print(input_table_items_list)
                    # print(type(input_table_items_list))
                    ###==============将遍历的元素添加到tablewidget中并显示=======================

                    input_table_items = str(input_table_items_list)
                    newItem = QTableWidgetItem(input_table_items)
                    newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    self.tableWidget.setItem(i, j, newItem)
                    ###================遍历表格每个元素，同时添加到tablewidget中========================
        else:
            self.centralWidget.show()

    def get_flagsKindNumber(self):  # 统计每一个pcap包的协议标识种类数
        kinds = []
        for f in flags:
            if f in kinds:
                continue
            else:
                kinds.append(f)
        return len(kinds) / 8

    def get_pkttranstimerate(self):  # 相邻数据包传送时间间隔均值
        f = open(path_openfile_name, 'rb')
        Timestamp = []
        pcap = dpkt.pcap.Reader(f)
        for timestamp, buf in pcap:
            Timestamp.append(timestamp)
        interval = 0
        for i in range(len(Timestamp) - 1):
            interval += Timestamp[i + 1] - Timestamp[i]
        if len(Timestamp) == 1:
            interval_mean = interval
        else:
            interval_mean = interval / (len(Timestamp) - 1)
        return Timestamp, interval_mean

    def get_pktpayloadsize(self):
        tmp = np.array(payload_size)
        return tmp.mean()

    def get_pkttransrate(self):  # t时间内从源ip到目的ip传送的packet数目
        t = Timestamp[len(Timestamp) - 1] - Timestamp[0]
        if t == 0:
            rate = 0
        else:
            rate = flow_size / t
        return rate

    def get_statistic_info(self):
        x_name = ["flags_kind", "flow_size", "pkttrans_timerate", "pkttrans_rate", "payload_size_mean"]
        output = pd.DataFrame(columns=x_name)
        csv_stat="./csv/data_stat.csv"
        output.to_csv(csv_stat, index=False)  # 把一个DataFrame写入csv文件
        pcap_file = path_openfile_name
        print(pcap_file)
        if pcap_file:
            sport = []
            dport = []
            global flags
            flags = []
            global payload_size
            payload_size = []
            global flow_size
            flow_size = 0
            packages = rdpcap(pcap_file)
            for data in packages:
                flow_size += 1
                if 'TCP' in data:
                    sport.append(data['TCP'].sport)
                    dport.append(data['TCP'].dport)
                    flags.append(data['TCP'].flags)
                    payload_size.append(len(data.payload))
                else:
                    sport.append('-1')
                    dport.append('-1')
                    flags.append('A')
                    payload_size.append(len(data.payload))
            # print(tmp)
            flags_kind = self.get_flagsKindNumber()
            # print("flags_kind:", flags_kind)
            global Timestamp
            Timestamp, pkttrans_timerate = self.get_pkttranstimerate()
            pkttrans_rate = self.get_pkttransrate()
            payload_size_mean = self.get_pktpayloadsize()

            feature = [flags_kind] + [flow_size] + [
                pkttrans_timerate] + [pkttrans_rate] + [
                          payload_size_mean]
            # feature = [flags_kind]
            f=open(csv_stat, 'a')
            f.write(str(feature).strip('[]') + '\n')

            global path_csv_stat
            path_csv_stat = csv_stat

    def creat_table_show2(self):
        ###===========读取表格，转换表格，===========================================
        if len(path_csv_stat) > 0:
            input_table = pd.read_csv(path_csv_stat)
            print(input_table)
            input_table_rows = input_table.shape[0]
            input_table_colunms = input_table.shape[1]
            print(input_table_rows)
            print(input_table_colunms)
            input_table_header = input_table.columns.values.tolist()
            print(input_table_header)

            ###===========读取表格，转换表格，============================================
            ###======================给tablewidget设置行列表头============================

            self.tableWidget_2.setColumnCount(input_table_colunms)
            self.tableWidget_2.setRowCount(input_table_rows)
            self.tableWidget_2.setHorizontalHeaderLabels(input_table_header)

            ###======================给tablewidget设置行列表头============================

            ###================遍历表格每个元素，同时添加到tablewidget中========================
            for i in range(input_table_rows):
                input_table_rows_values = input_table.iloc[[i]]
                # print(input_table_rows_values)
                input_table_rows_values_array = np.array(input_table_rows_values)
                input_table_rows_values_list = input_table_rows_values_array.tolist()[0]
                # print(input_table_rows_values_list)
                for j in range(input_table_colunms):
                    input_table_items_list = input_table_rows_values_list[j]
                    # print(input_table_items_list)
                    # print(type(input_table_items_list))
                    ###==============将遍历的元素添加到tablewidget中并显示=======================

                    input_table_items = str(input_table_items_list)
                    newItem = QTableWidgetItem(input_table_items)
                    newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    self.tableWidget_2.setItem(i, j, newItem)
                    ###================遍历表格每个元素，同时添加到tablewidget中========================
        else:
            self.centralWidget.show()

    def padArray(self):  #补齐
        array_ = packetPayload
        length=input_size
        if len(array_) > length:
            return array_[:length]  # 长度超过length则返回前length个元素
        else:
            return array_ + [-1] * (length - len(array_))  # 长度不足length则补全

    def get_payload_info(self):
        global pkt_counts
        global pkt_size
        global input_size
        pkt_counts = 18  # 从get_pktsize_meanOfEachPcap()中获得
        pkt_size = 64
        input_size = pkt_counts * pkt_size

        packets = rdpcap(path_openfile_name)
        # 返回负载特征
        payloads = ""
        global packetPayload
        packetPayload = []
        for pkt in packets[:pkt_counts]:  # 丢弃
            payloads += str(pkt.payload)[:pkt_size]
            packetPayload = [ord(c) for c in payloads]  # 转换成字符对应的ASCII码
            # minmax归一化
            packetPayload = self.padArray()  # 补齐

        x_name = ["x_%s" % str(i) for i in range(input_size)]
        output = pd.DataFrame(columns=x_name)
        csv_payload = "./csv/data_payload.csv"
        output.to_csv(csv_payload, index=False)  # 把一个DataFrame写入csv文件
        feature = packetPayload
        f = open(csv_payload, 'a')
        f.write(str(feature).strip('[]') + '\n')

        global path_csv_payload
        path_csv_payload=csv_payload

    def creat_table_show3(self):
        ###===========读取表格，转换表格，===========================================
        if len(path_csv_payload) > 0:
            input_table = pd.read_csv(path_csv_payload)
            print(input_table)
            input_table_rows = input_table.shape[0]
            input_table_colunms = input_table.shape[1]
            print(input_table_rows)
            print(input_table_colunms)
            input_table_header = input_table.columns.values.tolist()
            print(input_table_header)

            ###===========读取表格，转换表格，============================================
            ###======================给tablewidget设置行列表头============================

            self.tableWidget_3.setColumnCount(input_table_colunms)
            self.tableWidget_3.setRowCount(input_table_rows)
            self.tableWidget_3.setHorizontalHeaderLabels(input_table_header)

            ###======================给tablewidget设置行列表头============================

            ###================遍历表格每个元素，同时添加到tablewidget中========================
            for i in range(input_table_rows):
                input_table_rows_values = input_table.iloc[[i]]
                # print(input_table_rows_values)
                input_table_rows_values_array = np.array(input_table_rows_values)
                input_table_rows_values_list = input_table_rows_values_array.tolist()[0]
                # print(input_table_rows_values_list)
                for j in range(input_table_colunms):
                    input_table_items_list = input_table_rows_values_list[j]
                    # print(input_table_items_list)
                    # print(type(input_table_items_list))
                    ###==============将遍历的元素添加到tablewidget中并显示=======================

                    input_table_items = str(input_table_items_list)
                    newItem = QTableWidgetItem(input_table_items)
                    newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    self.tableWidget_3.setItem(i, j, newItem)
                    ###================遍历表格每个元素，同时添加到tablewidget中========================
        else:
            self.centralWidget.show()

    def dnncheck(self):
        # test===============================================
        # def func(s):
        #     str=s+"world"
        #     return str
        # str=func("hello")
        # self.textEdit_2.setText(str)
        # test===============================================

        print("Inputing data...")
        data = predict_1152()  # 导入检测数据
        x_detect = np.array(data)
        print("Inputing Finished")
        # print(x_detect)
        # print(x_detect.shape)

        sess = tf.Session()
        # 加载整个graph
        new_saver = tf.train.import_meta_graph('./model/ddos_detect_model.meta')
        # 加载模型中各种变量的值，注意这里不用文件的后缀
        new_saver.restore(sess, './model/ddos_detect_model')
        # print("1")
        # 对应第一个文件的add_to_collection()函数
        prediction = tf.get_collection('prediction')[0]  # 返回值是一个list，我们要的是第一个，这也说明可以有多个变量的名字一样。
        # print(prediction)
        # print("2")
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name('X').outputs[0]  # 将placeholder加载出来
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        # print(x)
        # print("3")
        pred = sess.run(prediction, feed_dict={x: x_detect, keep_prob: 0.7})
        print(str(pred))
        result=sess.run(tf.argmax(pred, 1))
        print(str(result))
        if result[0]==1:
            res="流量涉及DDoS攻击！"
        elif result[0]==0:
            res = "流量正常！"
        self.textEdit_2.setText(res)





if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())