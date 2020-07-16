#!/usr/bin/env python
#encoding=utf-8
from scapy.all import *
import numpy as np
import pandas as pd
import sys
import dpkt
#=================================================================
def print_progress(count, total):   #可视化转换进度
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)  #3.x版本中的用法 同print类似
    sys.stdout.flush()
#=================================================================

pkt_counts=16   #从get_pktsize_meanOfEachPcap()中获得
pkt_size=64
input_size=pkt_counts * pkt_size

# label<------> software name
softwares={
    "white":0,
    "ddos":1
}

# padding
def padArray(array_,num,length):
    if len(array_)>length:
        return array_[:length]   #长度超过length则返回前length个元素
    else:
        return array_+[num]*(length-len(array_))   #长度不足length则补全

def statistics(nums):
    if len(nums)>0:
        return [np.mean(nums),np.max(nums),np.min(nums),np.median(nums),np.var(nums)]
    else:
        return [0,0,0,0,0]


# ========================================================================
#01归一化 max min用np.max() np.min()求
def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min)
    return x

#sigmoid归一化
def sigmoid(X):
    return 1.0 / (1 + np.exp(-float(X)))

def packet_parse(pcap_file):
    '''
    解析pcap文件
    :param pcap_file:
    :return:payload,label
    '''
    next=0
    packets = rdpcap("./testing/"+pcap_file)

    #返回负载特征
    payloads = ""
    packetPayload=[]
    for pkt in packets[:pkt_counts]:  #丢弃
        if 'TCP' in pkt:
            # payloads+=str(pkt['IP'].src)
            # payloads+=str(pkt['IP'].dst)
            payloads += str(pkt['TCP'].sport)+' '
            payloads += str(pkt['TCP'].dport)+' '
            payloads += str(pkt['TCP'].flags)+' '
            payloads+=str(pkt.payload)[:pkt_size-3]+' '
        else:   #丢弃含UDP的packet
            payloads+="udp"
        packetPayload = [ord(c) for c in payloads]  # 转换成字符对应的ASCII码
        # tmp = []
        # for x in packetPayload:
        #     tmp.append(sigmoid(x))
        packetPayload = padArray(packetPayload, -1, pkt_counts * pkt_size)  # 补齐
    #返回标签特征
    i=pcap_file.find("ddos")   #返回“ddos”字符串的位置
    if i>=0:
        label=[1,0]  #ddos   one-hot编码
    else:
        label=[0,1] #ddos white

    return packetPayload,label

def get_Timestamp(pcap_file):   #得到该pcap文件中每一条的时间戳
    Timestamp=[]
    f = open("./testing/" + pcap_file,'rb')
    pcap = dpkt.pcap.Reader(f)
    for timestamp,buf in pcap:
        Timestamp.append(str(datetime.datetime.utcfromtimestamp(timestamp)))
    return Timestamp

# ==========================================================================

pcap_files=os.listdir("./testing")
#用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。
x_name = ["x_%s" % str(i) for i in range(input_size)]
x_name.append("label0")
x_name.append("label1")
output=pd.DataFrame(columns=x_name)
output.to_csv("./testing_data.csv",index=False)  #把一个DataFrame写入csv文件

def write_testdata():
    i=0
    num = len(pcap_files)
    x=[]  #输入特征
    y=[]  #标签
    f = open("./testing_data.csv", 'a')
    for pcap_file in pcap_files:
        i = i + 1
        print_progress(i, num)
        packetPayload,label = packet_parse(pcap_file)
        if "udp" in packetPayload:
            continue
        else:
            feature=packetPayload+label
            f.write(str(feature).strip('[]') + '\n')
    return x,y

write_testdata()