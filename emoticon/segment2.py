# -*- coding: utf-8 -*-

import jieba
import logging

def cut(line):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # jieba custom setting.
    jieba.set_dictionary('../jieba_dict/dict.txt.big')

    # load stopwords set
    stopword_set = set()
    with open('../jieba_dict/stopwords.txt','r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))

    words = jieba.cut(line, cut_all=False)
    
    resu = ""
    for word in words:
        if word not in stopword_set:
            resu += word + ' '
            
    return resu

#if __name__ == '__main__':
#    print(cut("回复@王羊羊的微博:是，要的就是安静的生存环境，而且，到北京五环25分钟车程"))
    
