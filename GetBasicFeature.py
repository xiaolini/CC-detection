from scapy.all import *
import pandas as pd
import os
import dpkt
import datetime

def get_Timestamp(pcap_file):   #得到该pcap文件中每一条的时间戳
    Timestamp=[]
    f = open("./train_data/" + pcap_file,'rb')
    pcap = dpkt.pcap.Reader(f)
    for timestamp,buf in pcap:
        Timestamp.append(str(datetime.datetime.utcfromtimestamp(timestamp)))
    return Timestamp

def get_Features(pcap_file):  #返回ip、port、flags、payload等基本信息
    src = []
    dst = []
    sport = []
    dport = []
    flags = []
    payload = []
    packages = rdpcap("./train_data/" + pcap_file)
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
    return src,dst,sport,dport,flags,payload

def get_payload(pcap_file):  # 返回负载特征  #返回一个n*80的DataFrame
    packages = rdpcap("./train_data/" + pcap_file)
    df = pd.DataFrame()
    for data in packages:
        # 返回负载特征
        payloads = str(data.payload)
        packetPayload = [ord(c) for c in payloads]  # 转换成字符对应的ASCII码
        df = df.append(pd.Series(packetPayload), ignore_index=True)
    return df

pcap_files = os.listdir("./train_data")  # 用于返回指定的文件夹包含的文件或文件夹的名字的列表
i = 0
for pcap_file in pcap_files:
    stop = 0

    file_name = "./csv/" + pcap_file + ".csv"
    sip,dip,sport,dport,flags,payload=get_Features(pcap_file)
    timestamp=get_Timestamp(pcap_file)
    d={'sip':sip,'dip':dip,'sport':sport,'dport':dport,'flags':flags,'time-stamp':timestamp,'payload':payload}
    # d = {'sip': sip, 'dip': dip, 'sport': sport, 'dport': dport, 'flags': flags}
    feature=pd.DataFrame(d)
    feature.to_csv(file_name, index=False)  # 把一个DataFrame写入csv文件
    i = i + 1
    print(i)



'''
pcap_files=os.listdir("./pcap")  #用于返回指定的文件夹包含的文件或文件夹的名字的列表
x_name = ["x_%s" % str(i) for i in range(7)]  #七个特征
x_name.append("label")
output=pd.DataFrame(columns=x_name)
output.to_csv("./test.csv",index=False)  #把一个DataFrame写入csv文件
for pcap_file in pcap_files:
    a = rdpcap("./pcap/"+pcap_file)
    print("打印每一个数据包中包含的协议数目:")
    print (a)
    print("读取第一个数据包的内容:")
    print (a[21])

    print("使用f[0].show()或f[0].display()进行数据包格式化打印:")
    print (a[21].display())
    for data in a:
        if 'TCP' in data:
            s = repr(data)
            print(s)
            print("源端口：", data['TCP'].sport)
            break

    #f[0]指第一条流
    #通过f[0].payload可以得到该层协议的载荷部分，
    #通过重复调用payload方法可以找到自己需要的层，并对其下的字段进行提取分析。 
    
    print(a[21].payload)
    print("\n")
    break
'''

