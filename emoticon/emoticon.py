# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 08:17:58 2020

@author: Tai
"""

import  xml.dom.minidom
import  re

  #打开xml文档
dom = xml.dom.minidom.parse('emoticon.xml')

  #得到文档元素对象
root = dom.documentElement

weibolist = root.getElementsByTagName('weibo')
weibolist1 = root.getElementsByTagName('sentence') #获得标签名
b= weibolist[0]
c= weibolist1[22]

un=c.getAttribute("emotion-1-type") #获得标签属性值

pattern = re.compile('\[.*\]')
str = '你好[无聊]'
#print(pattern.search(str))

#print(b.nodeName, c.nodeName, un, c.firstChild.data)
 
content = c.firstChild.data
regex = re.compile('\[[\u4e00-\u9fa5]*\]')
x = regex.search(content)
print(x.group())