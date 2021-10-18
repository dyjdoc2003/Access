# import torch
# import torch.nn as nn
# from GRU.Dynamic_RNN import DynamicRNN
#
#
# class GRU(nn.Module):
#     def __init__(self, args, embedding_matrix=None, aspect_embedding_matrix=None):
#         super(GRU, self).__init__()
#         self.args = args
#         self.encoder = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
#         self.encoder_aspect = nn.Embedding.from_pretrained(torch.tensor(aspect_embedding_matrix, dtype=torch.float))
#         self.gru = DynamicRNN(args.embed_dim, args.hidden_dim, num_layers=1, dropout=args.dropout, rnn_type="GRU")
#         self.dense = nn.Linear(args.hidden_dim, args.polarities_dim)
#         self.softmax = nn.Softmax()
#         self.dropout = nn.Dropout(args.dropout)
#
#     def forward(self, inputs):
#         text_raw_indices = inputs[0]
#         x = self.encoder(text_raw_indices)
#         x = self.dropout(x)
#         x_len = torch.sum(text_raw_indices != 0, dim=-1)
#         _, (h_n, _) = self.gru(x, x_len)
#         output = h_n[0]
#         output = self.dropout(output)
#         output = self.dense(output)
#         if self.args.softmax:
#             output = self.softmax(output)
#         return output

import  numpy as np
from random import randint
import tensorflow as tf
import datetime
import gensim
#---------------------------加载训练数据------------------------
wordsList = np.load('./training_data/wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()  # Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
# 加载词向量嵌入矩阵
wordVectors = np.load('./training_data/wordVectors.npy')
print('Loaded the word vectors!')
# 加载下标映射矩阵
ids = np.load('./training_data/idsMatrix.npy')
#--------------------------参数配置-----------------------------
maxSeqLength = 250
batchSize = 24 #批次大小，一次30张放入神经网络训练
gruUnits = 64#隐藏层单元的维度
numClasses = 2#二分类
numDimensions = 50 #Dimensions for each word vector，一个词50维
iterations = 100000 #2w次左右基本过拟合,7000左右acu=88.3%
#-------------------------获取训练批次数据-----------------------
def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1,11499)
            labels.append([1,0])  #负类情感
        else:
            num = randint(13499,24999)
            labels.append([0,1])  #正类情感
        # ids=loadTrainData()
        arr[i] = ids[num-1:num]
    return arr, labels

#-------------------------定义神经网络变量------------------------
tf.reset_default_graph()
#labels是类别标签:[0,1]表示正类，[1,0]表示负类
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
#在嵌入矩阵中根据id查找词的向量，组成该句话的句子张量
keep_prob = tf.placeholder(tf.float32)
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)
#-------------------------定义GRU神经网络------------------
gruCell = tf.contrib.rnn.GRUCell(gruUnits)
#dropout.每批数据输入时神经网络中的每个单元会以1 - keep_prob的概率不工作，可以防止过拟合
mgru_cell = tf.contrib.rnn.DropoutWrapper(cell=gruCell,output_keep_prob=0.75)
# mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([unit_lstm() for i in range(3)], state_is_tuple=True)#多层LSTM
#-------------------------定义神经网络变量---------------------------
value, _ = tf.nn.dynamic_rnn(mgru_cell, data, dtype=tf.float32)

#设置全连接层参数
#LSTM输出的向量的维度是指定的units，但是最后在计算损失的时候用的标签，标签的向量维度和units不一致，
#这样就没办法计算损失了，所以需要加一个全连接Dense将输出向量转换成标签向量的维度，这样就能进行损失计算了。
weight = tf.Variable(tf.truncated_normal([gruUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
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

#value结果进行转置
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
# with tf.name_scope('Attention_layer'):
outputs, _ = attention(value, 64, time_major=True)
print("attention outputs:", outputs)
prediction = (tf.matmul(outputs, weight) + bias)#全连接层
correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
#定义交叉熵损失和Adam优化器
#使用交叉熵+L2正则化损失来训练模型
#l2正则化
tf.add_to_collection(tf.GraphKeys.WEIGHTS, prediction)
regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
reg_term = tf.contrib.layers.apply_regularization(regularizer)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))
loss=loss+reg_term
#使用Adam优化器
optimizer = tf.train.AdamOptimizer().minimize(loss)

#-------------------------训练配置-----------------------------------
sess = tf.InteractiveSession()
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
#日志记录
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
#--------------------------训练模型-----------------------------------
with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        # Next Batch of reviews
        nextBatch, nextBatchLabels = getTrainBatch();

        sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

        # Write summary to Tensorboard
        if (i % 10 == 0):
            summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
            writer.add_summary(summary, i)
            acc = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})
            print('training! Step%d Training accuracy:%f' % (i, acc))

        # Save the network every 10,000 training iterations
        if (i % 10 == 0 and i != 0):
            save_path = saver.save(sess, "./models/pretrained_gru.ckpt", global_step=i)
            print("saved to %s" % save_path)
    writer.close()
