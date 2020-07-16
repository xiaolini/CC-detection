import pandas as pd
import numpy as np
def input_testdata():

    output_test=pd.read_csv("./testing_data.csv")
    common_features=["x_%d"%i for i in range(1152)]
    label=["label0","label1"]
    test_labels=np.asarray(output_test[label],dtype=np.float32)
    test_data=np.asarray(output_test[common_features],dtype=np.float32)
    # print (train_data.shape)
    # print (train_labels.shape)
    # print (train_labels)
    return test_data,test_labels
# inputs()