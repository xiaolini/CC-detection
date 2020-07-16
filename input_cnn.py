import pandas as pd
import numpy as np
def inputs_1152():

    output_train1152=pd.read_csv("./data_payload_raw.csv") #一共1152个特征+2个标签=25
    common_features=["x_%d"%i for i in range(1152)]
    label=["label0","label1"]
    train_labels=np.asarray(output_train1152[label],dtype=np.float32)
    train_data=np.asarray(output_train1152[common_features],dtype=np.float32)
    # print (train_data.shape)
    # print (train_labels.shape)
    # print (train_labels)
    return train_data,train_labels


def inputs_5():
    output_train5 = pd.read_csv("./data_stat_raw.csv")
    common_features5 = ["flags_kind", "flow_size", "pkttrans_timerate", "pkttrans_rate", "payload_size_mean"]
    label=["label0","label1"]
    train_labels=np.asarray(output_train5[label],dtype=np.float32)
    train_data=np.asarray(output_train5[common_features5],dtype=np.float32)
    # print (train_data.shape)
    # print (train_labels.shape)
    # print (train_labels)
    return train_data,train_labels

def inputs_1157():
    output_train5=pd.read_csv("./data_stat_raw.csv")
    common_features5=["flags_kind",  "flow_size","pkttrans_timerate","pkttrans_rate","payload_size_mean"]
    output_train1152=pd.read_csv("./data_payload_raw.csv")
    common_features1152=["x_%d"%i for i in range(1152)]
    label=["label0","label1"]
    output_train=output_train5[common_features5]+output_train1152[common_features1152]
    train_labels=np.asarray(output_train1152[label],dtype=np.float32)
    train_data=np.asarray(output_train,dtype=np.float32)
    # print (train_data.shape)
    # print (train_labels.shape)
    # print (train_labels)
    return train_data,train_labels


