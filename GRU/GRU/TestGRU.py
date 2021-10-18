#https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
wordsList = np.load('./training_data/wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('./training_data/wordVectors.npy')
print ('Loaded the word vectors!')

print("wordList:",wordsList)
print(wordVectors.shape)
print("wordVectors:",wordVectors)

maxSeqLength = 250

ids = np.load('./training_data/idsMatrix.npy')
print("ids:",ids)
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

batchSize = 24#24
gruUnits = 64
numClasses = 2
iterations = 100000
numDimensions = 50 #Dimensions for each word vector
import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
#
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)

data = tf.nn.embedding_lookup(wordVectors,input_data)
print("%"*80)
print(data.shape)
gruCell = tf.contrib.rnn.GRUCell(gruUnits)
#dropout.每批数据输入时神经网络中的每个单元会以1 - keep_prob的概率不工作，可以防止过拟合
mgru_Cell = tf.contrib.rnn.DropoutWrapper(cell=gruCell,output_keep_prob=0.75)


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
#动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出也是这种形状
value, h = tf.nn.dynamic_rnn(mgru_Cell, data, dtype=tf.float32)
weight = tf.Variable(tf.truncated_normal([gruUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2]) #注意这里输出需要转置  转换为时序优先的
outputs, _ = attention(value, 64, time_major=True)
print("*"*80)
print(int(value.get_shape()[0]) - 1)

last = tf.gather(value, int(value.get_shape()[0]) - 1) #The value of this parameter is 249
prediction = (tf.matmul(outputs, weight) + bias)
pred=tf.argmax(prediction, 1)
lab= tf.argmax(labels,1)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))



sess = tf.InteractiveSession()
saver = tf.train.Saver()
#两个参数，当前的会话，保存的模型
saver.restore(sess, './models/pretrained_gru.ckpt-10320')
#导入一些模型没有看见过的数据集
iterations = 83
sum=0
for i in range(iterations):
        nextBatch, nextBatchLabels = getTestBatch(i)
        acc=sess.run([accuracy], {input_data: nextBatch, labels: nextBatchLabels})
        # print(pred)
        # print(lab)
        print("Accuracy for this batch is:",acc)
        sum=sum+acc[0]
        # print(data[1,:,:].shape)
print('Final:%f'%(sum/83))

