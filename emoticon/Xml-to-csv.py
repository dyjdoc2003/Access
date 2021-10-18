# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:14:20 2020

@author: Tai
"""

import csv
import time 
try: 
    import xml.etree.cElementTree as ET 
except ImportError: 
    import xml.etree.ElementTree as ET 
import sys
import os
import re
namespaces = {'i': 'raml20.xsd'}
mr_dic={}
mr_list=[]
fdd_list=[]
def parseXml():
    try:
        tree=ET.parse('emoticon.xml')
        root=tree.getroot()
    except Exception:
        print("error with parse xml")
        sys.exit(1)
    #解析class="MR"
    for mr in root.iterfind('i:data/i:weibo', namespaces):
        mr_dic={}# 每次遍历时创建一个新的内存,如不创建，会覆盖字典值
        label = re.findall(mr.attrib['emotion-type1'],mr.attrib['emotion-type2'])
        mr_dic['label']="".join(label)
        for mrname in mr.iterfind('i:sentence',namespaces):
            if mrname.attrib['id'] == '1' :
                mr_dic['ques']=mrname.text
            if mrname.attrib['id'] == '2' :
                mr_dic['pinyinName']=mrname.text
        mr_list.append(mr_dic)
    print(mr_list)
    #解析class="LN_FDD"
    for fdd in root.iterfind('i:cmData/i:managedObject[@class="LN_FDD"]',namespaces):
        fdd_list1=[]
        fdd_id = re.findall(r"MR-(\d+)/",fdd.attrib['distName'])
        fdd_list1.append("".join(fdd_id))
        for act1xSrvcc in fdd.iterfind('i:p[@name="act1xSrvcc"]',namespaces):
                fdd_list1.append(act1xSrvcc.text)
        fdd_list.append(fdd_list1)
    print(fdd_list)
#关联合并
def mrMoreact1xSrvcc(mrbts,fddbts):
    for one_mrbts in mrbts:
        for one_fddbts in fddbts:#for..else..
            if one_mrbts['id'] in one_fddbts[0]:
                one_mrbts['act1xSrvcc'].append(one_fddbts[1])
            else:
                one_mrbts['act1xSrvcc']=['']
def createDictCsv(fileName,combine_list):
    keys = combine_list[0].keys()
    for tem in combine_list:
        tem['act1xSrvcc']=",".join(tem.get('act1xSrvcc',''))
    with open(fileName, 'w',newline='') as output_file:
        dict_writer = csv.DictWriter(output_file,list(keys),delimiter="|")
        dict_writer.writeheader()
        dict_writer.writerows(combine_list)
if __name__ == '__main__':
    parseXml()
    mr_list[0]['act1xSrvcc']=[]
    mrMoreact1xSrvcc(mr_list,fdd_list)
    print(mr_list)
    createDictCsv('people.csv',mr_list)