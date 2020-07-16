import tensorflow as tf
import numpy as np
from predict_input import predict_1152,predict_5,predict_1157
import time

print("Inputing data...")
data=predict_1152()  #导入检测数据
x_detect=np.array(data)
print("Inputing Finished")
print (x_detect)
print(x_detect.shape)

sess = tf.Session()
#加载整个graph
new_saver = tf.train.import_meta_graph('./model/ddos_detect_model.meta')
#加载模型中各种变量的值，注意这里不用文件的后缀
new_saver.restore(sess,'./model/ddos_detect_model')
print("1")
#对应第一个文件的add_to_collection()函数
prediction = tf.get_collection('prediction')[0] #返回值是一个list，我们要的是第一个，这也说明可以有多个变量的名字一样。
print(prediction)
print("2")
graph = tf.get_default_graph()
x = graph.get_operation_by_name('X').outputs[0]#将placeholder加载出来
keep_prob=graph.get_operation_by_name('keep_prob').outputs[0]
print(x)
print("3")
pred = sess.run(prediction,feed_dict = {x:x_detect,keep_prob:0.7})
print(str(pred))








