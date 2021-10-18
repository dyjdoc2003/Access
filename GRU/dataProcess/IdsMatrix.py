# 删除标点符号、括号、问号等，只留下字母数字字符
import re
from os import listdir
from os.path import isfile, join
import numpy as np


wordsList = np.load('../LSTM/training_data/wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
print(len(wordsList))
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

positiveFiles = ['../data/IMDB/positiveReviews/' + f for f in listdir('../data/IMDB/positiveReviews/') if
                 isfile(join('../data/IMDB/positiveReviews/', f))]
def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

maxSeqLength = 250
fname = positiveFiles[4] #Can use any valid index (not just 3)
with open(fname) as f:
    for lines in f:
        print(lines)
        # exit
firstFile = np.zeros((maxSeqLength), dtype='int32')
print(fname)
with open(fname) as f:
    indexCounter = 0
    line = f.readline()
    cleanedLine = cleanSentences(line)
    split = cleanedLine.split()
    for word in split:
        try:
            firstFile[indexCounter] = wordsList.index(word)
            print(word)
        except ValueError:
            firstFile[indexCounter] = 399999  # Vector for unknown words
        indexCounter = indexCounter + 1
print(firstFile)