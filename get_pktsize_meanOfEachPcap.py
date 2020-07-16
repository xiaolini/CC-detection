from scapy.all import *
import os

def print_progress(count, total):   #可视化转换进度
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)  #3.x版本中的用法 同print类似
    sys.stdout.flush()

def getMeanOfList(list):
    sum = 0
    for i in list:
        sum += i
    mean = sum / len(list)
    return mean

def get_pktsize_meanOfEachPcap():
    print("1")
    pcap_files=os.listdir("./test-data")  #用于返回指定的文件夹包含的文件或文件夹的名字的列表
    f = open("./size_lengthOfEachPcap.csv", 'a') # 写入.csv文件
    num = len(pcap_files)
    size=[]
    i=0
    for pcap_file in pcap_files:
        i+=1
        print_progress(i, num)
        flow = rdpcap("./test-data/" + pcap_file)   #一个pcap包即一条流

        packetLength = [pkt.len for pkt in flow]  #每一个packet的长度

        packetsize=len(packetLength)  #每条流中包含多少个packet
        # size.append(packetsize)
        feature=[packetsize]+packetLength
        f.write(str(feature).strip('[]') + '\n')
    # pktsize_mean=getMeanOfList(size)
    # print("******************************")
    # print (size)
    # print (pktsize_mean)   #由此确定选取每个pcap包的【pktsize_mean】个packet，多丢弃，少补-1
    #通过模型训练确定选取每个pcap包中的每个packet的前【pktlen_mean】个数据，多丢弃，少补-1





get_pktsize_meanOfEachPcap()