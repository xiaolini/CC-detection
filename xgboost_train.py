import xgboost as xgb
from sklearn.model_selection import train_test_split
from input_cnn import inputs
from sklearn import metrics
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,precision_recall_fscore_support
x0,y0=inputs()  #导入数据集
y=[]
for i in y0:
    if i[0]==1:
        y.append(1)  #ddos
    elif i[0]==0:
        y.append(0)   #white
print (y)

train_X,test_X,train_y,test_y = train_test_split(x0,y,test_size=1/3,random_state=0)
dtrain=xgb.DMatrix(train_X,label=train_y)
dtest=xgb.DMatrix(test_X)

params={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':4,
    'lambda':10,
    'subsample':0.75,
    'colsample_bytree':0.75,
    'min_child_weight':2,
    'eta': 0.025,
    'seed':0,
    'nthread':8,
     'silent':1}

watchlist = [(dtrain,'train')]

bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)

ypred=bst.predict(dtest)

# 设置阈值, 输出一些评价指标
y_pred = (ypred >= 0.5)*1

print ('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))
print ('Accuracy: %.4f' % metrics.accuracy_score(test_y,y_pred))
print ('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
print ('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
# metrics.confusion_matrix(test_y,y_pred)