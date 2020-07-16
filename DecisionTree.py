import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,precision_recall_fscore_support
from input_cnn import inputs
import matplotlib.pyplot as plt

x0,y0=inputs()  #导入数据集
y=[]
for i in y0:
    if i[0]==1:
        y.append(1)  #ddos
    elif i[0]==0:
        y.append(0)   #white
# print (y)

train_X,test_X,train_y,test_y = train_test_split(x0,y,test_size=1/3,random_state=1)	#这里划分数据以1/3的来划分 训练集训练结果 测试集测试结果

pipeline = Pipeline([
    ('clf', DecisionTreeClassifier(criterion='entropy'))
])

parameters = {
    'clf__max_depth': (155, 160, 165),
    'clf__min_samples_split': (3, 4, 5),
    'clf__min_samples_leaf': (2, 3, 4)
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='f1')
grid_search.fit(train_X, train_y)
# print('最佳效果：%0.3f' % grid_search.best_score_)
# print('最优参数')
# best_parameters = grid_search.best_best_parameters = grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print('\t%s: %r' % (param_name, best_parameters[param_name]))
y_pred = grid_search.predict(test_X)
print("accuracy_score = "+str(accuracy_score(test_y, y_pred)))  # 0.5
# normalize：bool型值，默认为True；如果为False，则表示正确分类的样本数
# print("accuracy_score = "+accuracy_score(test_y, y_pred, normalize=False))  # 4
# 1)average=None计算每个类的分数
print("precision_score = "+str(precision_score(test_y, y_pred, average="micro")))
print("recall_score = "+str(recall_score(test_y, y_pred, average="micro")))
print("f1_score = "+str(f1_score(test_y, y_pred, average="micro")))
# print(classification_report(test_y, y_pred))
