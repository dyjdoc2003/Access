# -*- coding: utf-8 -*-

from gensim.models import word2vec
from gensim import models
import logging

def main():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	model = models.Word2Vec.load('word2vec.model')

	print("提供 3 种测试模式\n")
	print("輸入一个词，则去寻找前一百个改词的相似词")
	print("輸入两个词，则去计算两个词的余弦相似度")
	print("輸入三个词，进行类比推理")

	while True:
		try:
			query = input()
			q_list = query.split()

			if len(q_list) == 1:
				print("相似詞前 100 排序")
				res = model.most_similar(q_list[0],topn = 100)
				for item in res:
					print(item[0]+","+str(item[1]))

			elif len(q_list) == 2:
				print("计算 Cosine 相似度")
				res = model.similarity(q_list[0],q_list[1])
				print(res)
			else:
				print("%s之於%s，如%s之於" % (q_list[0],q_list[2],q_list[1]))
				res = model.most_similar([q_list[0],q_list[1]], [q_list[2]], topn= 100)
				for item in res:
					print(item[0]+","+str(item[1]))
			print("----------------------------")
		except Exception as e:
			print(repr(e))

if __name__ == "__main__":
	main()
