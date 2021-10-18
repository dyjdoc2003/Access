# # -*- coding: utf-8 -*-
#
# """
# 功能：测试gensim使用
# 时间：2016年5月2日 18:00:00
# """
#
from gensim.models import word2vec
import logging

# # 主程序日志记录
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus("E:\\5120154230PythonCode\\MyFirstLSTM\\data\\IMDB\\text8")  # 加载语料
model = word2vec.Word2Vec(sentences, size=200)  # 训练skip-gram模型; 默认window=5
print("model:",model)
# 计算两个词的相似度/相关程度
print("the word vector for man is:",model['man'])
y1 = model.similarity("woman", "man")
print (u"woman和man的相似度为：", y1)
print ("---------------\n")

# 计算某个词的相关词列表
y2 = model.most_similar("good", topn=20)  # 20个最相关的
print (u"和good最相关的词有：\n")
for item in y2:
    print (item[0], item[1])
print ("---------------\n")

# 寻找对应关系
print (' "boy" is to "father" as "girl" is to ...? \n')
y3 = model.most_similar(['girl', 'father'], ['boy'], topn=3)
for item in y3:
    print (item[0], item[1])
print ("----------------\n")

more_examples = ["he his she", "big bigger bad", "going went being"]
for example in more_examples:
    a, b, x = example.split()
    predicted = model.most_similar([x, b], [a])[0][0]
    print ("'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted))
print ("----------------\n")

# 寻找不合群的词
y4 = model.doesnt_match("breakfast cereal dinner lunch".split())
print (u"不合群的词：", y4)
print ("----------------\n")

# 保存模型，以便重用
model.save("text8.model")
# 对应的加载方式
# model_2 = word2vec.Word2Vec.load("text8.model")

# 以一种C语言可以解析的形式存储词向量
model.wv.save_word2vec_format("text8.model.bin", binary=True)
# 对应的加载方式
# model_3 = word2vec.Word2Vec.load_word2vec_format("text8.model.bin", binary=True)

if __name__ == "__main__":
    pass
import numpy as np
word2Vec=np.load("text8.model.wv.vectors.npy")
print(word2Vec)
wordList=np.load("text8.model")
print(wordList)
###自注意力机制torch
# import torch
#
# # 1、准备输入：Input 1、2、3
# x = [[1, 0, 1, 0],
#      [0, 2, 0, 2],
#      [1, 1, 1, 1]]
# x = torch.tensor(x, dtype=torch.float32)
#
# # 2、初始化权重
# w_key = [[0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0]]
# w_query = [[1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1]]
# w_value = [[0, 2, 0], [0, 3, 0], [1, 0, 3], [1, 1, 0]]
#
# w_key = torch.tensor(w_key, dtype=torch.float32)
# w_query = torch.tensor(w_query, dtype=torch.float32)
# w_value = torch.tensor(w_value, dtype=torch.float32)
#
# # 3、推导键、查询和值
# keys = x @ w_key
# querys = x @ w_query
# values = x @ w_value
#
# print(keys)  # tensor([[0., 1., 1.], [4., 4., 0.], [2., 3., 1.]])
# print(querys)  # tensor([[1., 0., 2.], [2., 2., 2.], [2., 1., 3.]])
# print(values)  # tensor([[1., 2., 3.], [2., 8., 0.], [2., 6., 3.]])
#
# # 4、计算注意力得分
# attn_scores = querys @ keys.t()
# # tensor([[ 2.,  4.,  4.],  # attention scores from Query 1
# #         [ 4., 16., 12.],  # attention scores from Query 2
# #         [ 4., 12., 10.]]) # attention scores from Query 3
#
#
# # 5、计算softmax
# from torch.nn.functional import softmax
#
# attn_scores_softmax = softmax(attn_scores, dim=-1)
# print('attn_scores_softmax：', '\n', attn_scores_softmax)
# # tensor([[6.3379e-02, 4.6831e-01, 4.6831e-01],
# #         [6.0337e-06, 9.8201e-01, 1.7986e-02],
# #         [2.9539e-04, 8.8054e-01, 1.1917e-01]])
#
# # For readability, approximate the above as follows
# attn_scores_softmax = [[0.0, 0.5, 0.5], [0.0, 1.0, 0.0], [0.0, 0.9, 0.1]]
# attn_scores_softmax = torch.tensor(attn_scores_softmax)
#
# # 6、将得分和值相乘
# weighted_values = values[:, None] * attn_scores_softmax.t()[:, :, None]
# print('weighted_values：', '\n', weighted_values)
# # tensor([[[0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000]],
# #         [[1.0000, 4.0000, 0.0000], [2.0000, 8.0000, 0.0000], [1.8000, 7.2000, 0.0000]],
# #         [[1.0000, 3.0000, 1.5000], [0.0000, 0.0000, 0.0000], [0.2000, 0.6000, 0.3000]]])
#
# # 7、求和加权值
# outputs = weighted_values.sum(dim=0)
# # tensor([[2.0000, 7.0000, 1.5000], [2.0000, 8.0000, 0.0000], [2.0000, 7.8000, 0.3000]]) # Output1、2、3
# print('outputs：', '\n', outputs)



