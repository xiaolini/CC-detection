import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from predict_input import predict_1152,predict_5,predict_1157
import time
from input_cnn import inputs_1152,inputs_5,inputs_1157

print("Importing dataset...")
x0,y0=inputs_1152()  #导入训练数据集1152
# x0,y0=inputs_5() #导入训练数据集5
# x0,y0=inputs_1157()  #导入训练数据集1157
data=predict_1152()  #导入检测数据
x_detect=np.array(data)
print("Finished")

#分成训练集和测试集 #这里划分数据以1/3的来划分 test_size指测试集占比
train_X,test_X,train_y,test_y = train_test_split(x0,y0,test_size=1/3,random_state=1)
train_X=np.array(train_X)
train_y=np.array(train_y)
[n,x_dim]=train_X.shape
print(x_dim)
# print(train_X.shape)
# print(test_X)
# print(train_y.shape)
# print(test_y)

#每个批次的大小
batch_size = 20
#计算一共有多少个批次
n_batch = len(train_X) // batch_size   #返回一个不大于结果的最大整数

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,x_dim],name = 'X')  #x列 行不确定
y = tf.placeholder(tf.float32,[None,2],name = 'Y')  #是ddos=1.或 不是ddos=0

keep_prob = tf.placeholder(tf.float32,name='keep_prob')

#创建一个简单的神经网络  包含两个隐藏层
W1 = tf.Variable(tf.truncated_normal([x_dim,500],stddev=0.1))#二初始值
b1 = tf.Variable(tf.zeros([500])+0.1)
L1 = tf.nn.relu(tf.matmul(x,W1)+b1)
L1_drop = tf.nn.dropout(L1,keep_prob)

W2 = tf.Variable(tf.truncated_normal([500,300],stddev=0.1))#二初始值
b2 = tf.Variable(tf.zeros([300])+0.1)
L2 = tf.nn.relu(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.truncated_normal([300,2],stddev=0.1))#二初始值
b3 = tf.Variable(tf.zeros([2])+0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop,W3)+b3)

#交叉熵代价函数
# cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=prediction))
# 二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))  #三交叉熵

#使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(0.00001).minimize(loss)

#结果存放在一个布尔列表中
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1)) #argmax返回一维张量中最大的值所在的位置
#求训练准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#函数tf.argmax()返回值是是数值最大值的索引位置，如果最大值位置相同，则分类正确，反之则分类错误
predictions = tf.argmax(prediction, 1)
actuals = tf.argmax(y, 1)
#将上述代码获得到变量设置为元素为0或者1的矩阵，在后面计算的时候只需要按照逻辑与计算即可
ones_like_actuals = tf.ones_like(actuals)
zeros_like_actuals = tf.zeros_like(actuals)
ones_like_predictions = tf.ones_like(predictions)
zeros_like_predictions = tf.zeros_like(predictions)

tp_op = tf.reduce_sum(
    tf.cast(
        tf.logical_and(
            tf.equal(actuals, ones_like_actuals),
            tf.equal(predictions, ones_like_predictions)
        ),
        "float"
    )
)

tn_op = tf.reduce_sum(
    tf.cast(
        tf.logical_and(
            tf.equal(actuals, zeros_like_actuals),
            tf.equal(predictions, zeros_like_predictions)
        ),
        "float"
    )
)

fp_op = tf.reduce_sum(
    tf.cast(
        tf.logical_and(
            tf.equal(actuals, zeros_like_actuals),
            tf.equal(predictions, ones_like_predictions)
        ),
        "float"
    )
)

fn_op = tf.reduce_sum(
    tf.cast(
        tf.logical_and(
            tf.equal(actuals, ones_like_actuals),
            tf.equal(predictions, zeros_like_predictions)
        ),
        "float"
    )
)

def splitdataset(list,batch):
    return list[batch_size*batch:batch_size*(batch+1)]

with tf.Session() as sess:
    start = time.clock()  # 开始计时
    sess.run(tf.global_variables_initializer())
    m_saver = tf.train.Saver()  # 模型保存

    for batch in range(n_batch):
       # batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs=splitdataset(train_X, batch)
        batch_ys = splitdataset(train_y, batch)
        sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
    # acc = sess.run(accuracy, feed_dict={x: test_X, y: test_y, keep_prob: 1.0})
    # print("Training：Iter " + str(epoch) + ", Accuracy= " + str(acc))
    print("prediction:",sess.run(tf.argmax(prediction,1),feed_dict={x:x_detect,keep_prob:0.7}))
    # 假如想要保存hypothesis和cost，以便在保存模型后，重新导入模型时可以使用。
    tp, tn, fp, fn = sess.run([tp_op, tn_op, fp_op, fn_op], feed_dict={x: test_X, y: test_y, keep_prob: 0.7})
    tpr = float(tp) / (float(tp) + float(fn))
    fpr = float(fp) / (float(fp) + float(tn))
    fnr = float(fn) / (float(tp) + float(fn))
    accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))
    recall = float(tp) / (float(tp) + float(fn))
    precision = float(tp) / (float(tp) + float(fp))
    f1_score = (2 * (precision * recall)) / (precision + recall)
    print("accuracy= " + str(accuracy) + ", recall= " + str(recall) + ", precision= " + str(
        precision) + ", f1_score= " + str(f1_score))
    tf.add_to_collection('prediction', prediction)  # 必须有个名字，即第一个参数
    tf.add_to_collection('loss', loss)
    m_saver.save(sess, './model/ddos_detect_model')
    elapsed = (time.clock() - start)  # 结束计时
    print("training time:" + str(elapsed) + "s")  # 印出时间
