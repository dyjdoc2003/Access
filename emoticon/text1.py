# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:14:20 2020

@author: Tai
"""

import csv
try: 
    import xml.etree.cElementTree as ET 
except ImportError: 
    import xml.etree.ElementTree as ET 
import sys
import segment2

def parseXml():
    try:
        tree=ET.parse('nlpcc2013.xml')
        root=tree.getroot()
    except Exception:
        print("error with parse xml")
        sys.exit(1)
    resu = []
    for mr in root.iterfind('.//weibo'):
        emotion_type = "" if 'emotion-type' not in mr.attrib.keys() else mr.attrib['emotion-type']
        emotion_type1 = "" if 'emotion-type1' not in mr.attrib.keys() else mr.attrib['emotion-type1']
        emotion_type2 = "" if 'emotion-type2' not in mr.attrib.keys() else mr.attrib['emotion-type2']
            
        sentences = ""
        for mrname in mr.iterfind('.//sentence'):
            sentences += mrname.text
        #jieba
        resu.append([emotion_type, emotion_type1, emotion_type2, segment2.cut(sentences)])
    
    return resu
   

def createDictCsv():
    filename = "nlpcc2013.csv"
    temp = parseXml()
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, lineterminator="\n")
        for item in temp:
            spamwriter.writerow(item)
    print("SUCCESS!")
        
if __name__ == '__main__':
    createDictCsv()
    