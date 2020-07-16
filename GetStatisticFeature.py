from scapy.all import *
import pandas as pd
import os
import dpkt
import time
import numpy as np

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

def get_flagsKindNumber(flags): #统计每一个pcap包的协议标识种类数

    kinds=[]
    for f in flags:

        if f in kinds:
            continue
        else:
            kinds.append(f)
    return len(kinds)/8

def get_portKindNumber(sport,dport):#统计每一个pcap包的端口号数目
    kinds=[]
    for p in sport:
        if p in kinds:
            continue
        kinds.append(p)
    for p in dport:
        if p in kinds:
            continue
        kinds.append(p)
    return len(kinds)

def get_TranSpeed(tmp):   #返回对流比
    b=tmp['sip']
    a=tmp['dip']
    d={}

    for i in range(len(b)):
        c1 = b[i] + a[i]
        if c1 in d:
            d[c1] = d[c1] + 1
        else:
            d[c1] = 1
   # print(d)
    for i in range(len(b)):
        c1 = b[i] + a[i]
        c2 = a[i] + b[i]
        if c1 in d and d[c1] == 0:
            del d[c1]
        if c2 in d and d[c2] == 0:
            del d[c2]
        if c1 in d and c2 in d:
            d[c1] = d[c1] - 1
            d[c2] = d[c2] - 1
    #print (d)
    sum = 0
    for key, value in d.items():
        if value > 0:
            sum = sum + value
    #print(sum)
    kind = (len(b) - sum) / 2
    return kind*2/len(b)


def get_pkttransrate(Timestamp,flow_size):#t时间内从源ip到目的ip传送的packet数目
    t=Timestamp[len(Timestamp)-1]-Timestamp[0]
    if t==0:
        rate=0
    else:
        rate=flow_size/t
    return rate


def get_pkttranstimerate(filename):#相邻数据包传送时间间隔均值
    f = open(filename, 'rb')
    Timestamp = []
    pcap = dpkt.pcap.Reader(f)
    for timestamp, buf in pcap:
        Timestamp.append(timestamp)
    interval = 0
    for i in range(len(Timestamp) - 1):
        interval += Timestamp[i + 1] - Timestamp[i]
    if len(Timestamp)==1:
        interval_mean = interval
    else:
        interval_mean = interval / (len(Timestamp) - 1)
    return Timestamp,interval_mean

def get_pktpayloadsize(payload_size):
    tmp=np.array(payload_size)
    return tmp.mean()


def get_StatisticFeatures():
    # 用于返回指定的文件夹包含的文件或文件夹的名字的列表
    start=time.clock()
    pcap_files=os.listdir("./train_data")
    i=0
    num = len(pcap_files)
    x_name = ["flags_kind",  "flow_size","pkttrans_timerate","pkttrans_rate","payload_size_mean"]
    x_name.append("label0")
    x_name.append("label1")
    output = pd.DataFrame(columns=x_name)
    output.to_csv("data_stat_raw.csv", index=False)
    # output.to_csv("data_stat_normalise.csv", index=False)  # 把一个DataFrame写入csv文件


    for pcap_file in pcap_files:
        i = i + 1
        print_progress(i, num)
        filename="./train_data/" + pcap_file
         # 一共七个基本信息
        # src = []
        # dst = []
        sport = []
        dport = []
        flags = []
        payload_size = []
        flow_size=0
        packages = rdpcap("./train_data/" + pcap_file)
        for data in packages:
            flow_size+=1
            if 'TCP' in data:
                # src.append(data['IP'].src)
                # dst.append(data['IP'].dst)
                sport.append(data['TCP'].sport)
                dport.append(data['TCP'].dport)
                flags.append(data['TCP'].flags)
                payload_size.append(len(data.payload))
            else:
                # src.append(data['IP'].src)
                # dst.append(data['IP'].dst)
                sport.append('-1')
                dport.append('-1')
                flags.append('A')
                payload_size.append(len(data.payload))
       # print(tmp)
        flags_kind = get_flagsKindNumber(flags)
        # print("flags_kind:", flags_kind)
        #port_Kind = get_portKindNumber(sport,dport)
        # print("port_Kind:", port_Kind)
        # pairflow_rate = get_TranSpeed(tmp)
        #print("pairflow_rate:", pairflow_rate)

        # print("flow_size:",flow_size)
        Timestamp,pkttrans_timerate=get_pkttranstimerate(filename)
        pkttrans_rate = get_pkttransrate(Timestamp,flow_size)
        payload_size_mean = get_pktpayloadsize(payload_size)
        if "ddos" in pcap_file:
            label=[1,0]  # 标记为包含ddos攻击的pcap
        else:
            label = [0, 1]  # 标记为正常的pcap

        feature = [flags_kind] + [flow_size] + [
            pkttrans_timerate] + [pkttrans_rate] + [
                      payload_size_mean] + label
        f = open("data_stat_raw.csv", 'a')
        f.write(str(feature).strip('[]') + '\n')

        # #归一化 sigmoid
        # # 1 / (1 + np.exp(-x))
        # feature = [1 / (1 + np.exp(-flags_kind))]+[1 / (1 + np.exp(-flow_size))]+[1 / (1 + np.exp(-pkttrans_timerate))]+[1 / (1 + np.exp(-pkttrans_rate))]+[1 / (1 + np.exp(-payload_size_mean))]+label
        # f=open("data_stat_normalise.csv", 'a')
        # f.write(str(feature).strip('[]') + '\n')
    tran_time=time.clock()-start
    print(tran_time)


get_StatisticFeatures()
