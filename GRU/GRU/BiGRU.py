import numpy as np
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
numDimensions = 300 #Dimensions for each word vector，一个词50维
iterations = 20000 #2w次左右基本过拟合
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
# with tf.name_scope('name'):
#-------------------------定义神经网络变量------------------------
tf.reset_default_graph()
#labels是类别标签:[0,1]表示正类，[1,0]表示负类,定义两个占位符，作为神经网络输入，占位符相当于形参
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
#-------------------------生成句子张量batchsize*maxSeqLength*numDimensions-----
#在嵌入矩阵中根据id查找词的向量，组成该句话的句子张量
keep_prob = tf.placeholder(tf.float32)
#变量的定义和初始化是分开的，一开始，tf.Variable 得到的是张量，而张量并不是具体的值，而是计算过程。
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

#value结果进行转置
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
# with tf.name_scope('Attention_layer'):
outputs, _ = attention(value, 64, time_major=True)
print("attention outputs:", outputs,outputs.shape)
# prediction = (tf.matmul(outputs, weight) + bias)#全连接层
#设置全连接层参数
weight = tf.Variable(tf.truncated_normal([gruUnits*2, numClasses])) #注意这里的维度
#创建0维常量tf.constant，tf.constant(-1.0, shape=[2, 3], name='b')
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
prediction=tf.matmul(tf.reshape(outputs, [-1, 2 * gruUnits]), weight) + bias  #注意这里的维度


correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
#定义交叉熵损失和Adam优化器
#使用交叉熵+L2正则化损失来训练模型
#l2正则化
tf.add_to_collection(tf.GraphKeys.WEIGHTS, prediction)
regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
reg_term = tf.contrib.layers.apply_regularization(regularizer)
#交叉熵
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))
loss=loss+reg_term
#使用Adam优化器
optimizer = tf.train.AdamOptimizer().minimize(loss)

#-------------------------训练配置-----------------------------------
sess = tf.InteractiveSession()
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()

logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
#--------------------------训练模型-----------------------------------
#首先对变量进行初始化，使用会话进行计算
with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    print("Start learning.......")
    for i in range(iterations):
        # Next Batch of reviews
        nextBatch, nextBatchLabels = getTrainBatch()

        sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

        # Write summary to Tensorboard
        if (i % 100 == 0): #接近一个epoch
            summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
            writer.add_summary(summary, i)
            loss_ = sess.run(loss, {input_data: nextBatch, labels: nextBatchLabels})
            acc = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})
            print('training! Step%d Training accuracy:%f' % (i, acc))
            print('train Step%d Training loss:%f' %(i,loss_))
            if(i!=0):
                save_path = saver.save(sess, "./bimodels/pretrained_gru.ckpt", global_step=i)
                print("saved to %s" % save_path)

    writer.close()
