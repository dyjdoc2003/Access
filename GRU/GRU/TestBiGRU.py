# #https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow
#
'''
# 设定超参数（隐藏向量的维度，学习率，输入输出维度等）
# 定义输入输出placeholder
# 定义网络结构，写好网络计算语句（dynamic_run等，用于计算时生成LSTM单元的输出状态）
# 定义全连接层的权重和偏差，用于将LSTM单元状态的输出转换成类别未规范化概率
# 计算输出的未规范化概率
# 定义softmax层
# 定义损失
# 定义训练优化器和优化操作
'''
import numpy as np
import os
from sklearn.metrics import f1_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
wordsList = np.load('./training_data/wordsList.npy')
print('Loaded the wordlList:{}'.format(wordsList))
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('./training_data/wordVectors.npy')
print ('Loaded the word vectors:{}'.format(wordVectors.shape))

print(len(wordsList))
print(wordVectors.shape)



ids = np.load('./training_data/idsMatrix.npy')
from random import randint

def getTestBatch(i1):
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        # num = randint(11499,13499)
        num=11499+i+batchSize*i1
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels
# def getTestBatch():
#     labels = []
#     arr = np.zeros([batchSize, maxSeqLength])
#     for i in range(batchSize):
#         num = randint(11499,13499)
#         if (num <= 12499):
#             labels.append([1,0])
#         else:
#             labels.append([0,1])
#         arr[i] = ids[num-1:num]
#     return arr, labels
def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels
maxSeqLength = 250
batchSize = 24#24
gruUnits = 64
numClasses = 2
numDimensions = 300 #Dimensions for each word vector
import tensorflow as tf
#-------------------------定义神经网络变量------------------------
tf.reset_default_graph()
#labels是类别标签:[0,1]表示正类，[1,0]表示负类,定义两个占位符，作为神经网络输入
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
#-------------------------生成句子张量batchsize*maxSeqLength*numDimensions-----
#在嵌入矩阵中根据id查找词的向量，组成该句话的句子张量
keep_prob = tf.placeholder(tf.float32)
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)
print("一个batch:",data.shape)
#dropout保存节点数，防止节点过多过拟合
#----------------------------双向GRU------------------------------------
fw_cells=tf.contrib.rnn.GRUCell(gruUnits)
fw_cells=tf.contrib.rnn.DropoutWrapper(cell=fw_cells, output_keep_prob=0.75)
bw_cells=tf.contrib.rnn.GRUCell(gruUnits)
bw_cells = tf.contrib.rnn.DropoutWrapper(cell=bw_cells, output_keep_prob=0.75)

init_fw=fw_cells.zero_state(batchSize,dtype=tf.float32)
init_bw=bw_cells.zero_state(batchSize,dtype=tf.float32)
outputs,final_states=tf.nn.bidirectional_dynamic_rnn(fw_cells,bw_cells,data,initial_state_fw=init_fw,initial_state_bw=init_bw)
value=tf.concat(outputs,2) #将前向和后向的状态连接起来
#-------------------------定义神经网络变量---------------------------
# value, _ = tf.nn.dynamic_rnn(mlstm_cell, data, dtype=tf.float32)

#------------------------注意力机制作用于最后一层------------------------------------------
def attention(inputs, attention_size, time_major=False):
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)
    if time_major:  # (T,B,D) => (B,T,D)
        inputs = tf.transpose(inputs, [1, 0, 2])
    hidden_size = inputs.shape[2].value
    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape
    # the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    return output, alphas
import torch
from torch.nn.functional import softmax
def selfAttention(inputs):
    x = torch.tensor(inputs, dtype=torch.float32)

#value结果进行转置
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
# with tf.name_scope('Attention_layer'):
outputs, _ = attention(value, 64, time_major=True)
print("attention outputs:", outputs,outputs.shape)
# prediction = (tf.matmul(outputs, weight) + bias)#全连接层
#设置全连接层参数
weight = tf.Variable(tf.truncated_normal([gruUnits*2, numClasses])) #注意这里的维度
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
prediction=tf.matmul(tf.reshape(outputs, [-1, 2 * gruUnits]), weight) + bias  #注意这里的维度

#函数tf.argmax()返回值是是数值最大值的索引位置，如果最大值位置相同，则分类正确，反之则分类错误
pred=tf.argmax(prediction, 1)

lab= tf.argmax(labels,1)
#---------------------------------------------

ones_like_actuals = tf.ones_like(lab)
zeros_like_actuals = tf.zeros_like(lab)
ones_like_predictions = tf.ones_like(pred)
zeros_like_predictions = tf.zeros_like(pred)

tp_op = tf.reduce_sum(
    tf.cast(
        tf.logical_and(
            tf.equal(lab, ones_like_actuals),
            tf.equal(pred, ones_like_predictions)#zeros_like_predictions解决分类，不只是二分类
        ),
        "float"
    )
)

tn_op = tf.reduce_sum(
    tf.cast(
        tf.logical_and(
            tf.equal(lab, zeros_like_actuals),
            tf.equal(pred, zeros_like_predictions)
        ),
        "float"
    )
)

fp_op = tf.reduce_sum(
    tf.cast(
        tf.logical_and(
            tf.equal(lab, zeros_like_actuals),
            tf.equal(pred, ones_like_predictions)
        ),
        "float"
    )
)

fn_op = tf.reduce_sum(
    tf.cast(
        tf.logical_and(
            tf.equal(lab, ones_like_actuals),
            tf.equal(pred, zeros_like_predictions)
        ),
        "float"
    )
)

#---------------------------------------------

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))



sess = tf.InteractiveSession()
saver = tf.train.Saver()
#两个参数，当前的会话，保存的模型
saver.restore(sess, './bimodels/pretrained_gru.ckpt-4300')
#one epoch=numbers of iterations=N=样本数量/batchSize
iterations = 83
sumAcc=sumF1=sumRecall=sumPre=0
sum=0
for i in range(iterations):
        nextBatch, nextBatchLabels = getTestBatch()
        acc=sess.run([accuracy], {input_data: nextBatch, labels: nextBatchLabels})
        tp, tn, fp, fn = sess.run([tp_op, tn_op, fp_op, fn_op], feed_dict={input_data: nextBatch, labels: nextBatchLabels})
        tpr = float(tp) / (float(tp) + float(fn))
        fpr = float(fp) / (float(fp) + float(tn))
        fnr = float(fn) / (float(tp) + float(fn))
        accur = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))
        recall = tpr
        precision = float(tp) / (float(tp) + float(fp))
        f1_score = (2 * (precision * recall)) / (precision + recall)
        print("This batch precision is:",precision)
        print("This batch recall is:",recall)
        print("This batch f1_score is:", f1_score)
        print("This batch accur is:",accur)
        print("This batch accuracy is",acc[0])
        sumAcc+=accur
        sumF1+=f1_score
        sumPre+=precision
        sumRecall+=recall
        sum+=acc[0]
        print(data[1,:,:].shape)
print("Final acc is :%f"%(sum/iterations))
print('Final ACU is:%f'%(sumAcc/iterations))
print('Final Pre is:%f'%(sumPre/iterations))
print('Final F1_score is:%f'%(sumF1/iterations))
print('Final Recall is:%f'%(sumRecall/iterations))
