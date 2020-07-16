import pandas as pd
import numpy as np
def predict_1152():

    output_predict1152=pd.read_csv("./csv/data_payload.csv")
    common_features=["x_%d"%i for i in range(1152)]
    predict_data=np.asarray(output_predict1152[common_features],dtype=np.float32)
    # print (train_data.shape)
    # print (train_labels.shape)
    # print (train_labels)
    return predict_data
# inputs()

def predict_5():
    output_predict5 = pd.read_csv("./csv/data_stat.csv")
    common_features5 = ["flags_kind", "flow_size", "pkttrans_timerate", "pkttrans_rate", "payload_size_mean"]

    predict_data=np.asarray(output_predict5[common_features5],dtype=np.float32)
    # print (train_data.shape)
    # print (train_labels.shape)
    # print (train_labels)
    return predict_data

def predict_1157():
    output_predict5=pd.read_csv("./csv/data_stat.csv")
    common_features5=["flags_kind",  "flow_size","pkttrans_timerate","pkttrans_rate","payload_size_mean"]
    output_predict1152=pd.read_csv("./csv/data_payload.csv")
    common_features1152=["x_%d"%i for i in range(1152)]

    output_train=output_predict5[common_features5]+output_predict1152[common_features1152]

    predict_data=np.asarray(output_train,dtype=np.float32)
    # print (train_data.shape)
    # print (train_labels.shape)
    # print (train_labels)
    return predict_data


