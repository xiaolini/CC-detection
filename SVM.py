import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from input_cnn import inputs
import time
import matplotlib.pyplot as plt

x0,y0=inputs()  #导入数据集
y=[]
for i in y0:
    if i[0]==1:
        y.append(1)  #ddos
    elif i[0]==0:
        y.append(0)   #white
#print (y)

train_X,test_X,train_y,test_y = train_test_split(x0,y,test_size=1/3,random_state=1)	#这里划分数据以1/3的来划分 训练集训练结果 测试集测试结果
c_range = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]  #取0-1间的30个随机小数
cv_scores = []		#用来放每个模型的结果值
for n in c_range:
    # clf = svm.SVC(C=n, kernel='linear')   #SVC模型，超参数C
    print ("1")
    start = time.clock()
    clf = svm.SVC(C=n, kernel='rbf',decision_function_shape='ovr')
    print ("2")
    clf.fit(train_X, train_y)
    train_time = (time.clock() - start)
    print("3")
    start2 = time.clock()
    y_pred = clf.predict(test_X)
    test_time = (time.clock() - start2)
    print (n)
    print("accuracy_score = "+str(accuracy_score(test_y, y_pred)))  # 0.5
    # normalize：bool型值，默认为True；如果为False，则表示正确分类的样本数
    # print("accuracy_score = "+accuracy_score(test_y, y_pred, normalize=False))  # 4
    # 1)average=None计算每个类的分数
    print("precision_score = "+str(precision_score(test_y, y_pred, average="micro")))
    print("recall_score = "+str(recall_score(test_y, y_pred, average="micro")))
    print("f1_score = "+str(f1_score(test_y, y_pred, average="micro")))
    print("train_time:" + str(train_time))
    print("test_time:" + str(test_time))




