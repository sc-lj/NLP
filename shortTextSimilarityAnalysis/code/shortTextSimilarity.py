# coding:utf-8

"""
基于论文A Method for Measuring Sentence Similarity and its Application to Conversational Agents中的方法计算短文本的相似度
"""

import os,re,jieba
import numpy as np
from TextProcess import *


def semanticSimilarity(sentence1,sentence2):
    """
    计算语句的语义相似度
    :param sentence1:
    :param sentence2:
    :return:
    """
    sentence1=cutWords(sentence1)
    sentence2=cutWords(sentence2)
    sentenset=list(set(sentence1).union(set(sentence2)))
    sentence1vector=[]
    sentence2vector=[]


# 返回格式{code: '文字字符串', }
def cilin():
    print('------当前进行《同义词词林》的读入操作------')
    cilinFilePath = '../script/synonym_ex.txt'
    cilinFileObject = open(cilinFilePath, 'r')  # 进行分词文件的读取
    cilinDatas = {}
    for line in cilinFileObject:
        word = line.strip('\n')  # 去除换行符
        cilinDatas[word[0:8]] = word[9:].split(" ")
    return cilinDatas

cilinDatas=cilin()
def wordEncoding(word,cilinDatas=cilinDatas):
    """
    从同义词林中找到词的编码
    :param word:
    :param cilinDatas:
    :return:
    """
    wordEncodingDatas=[]
    for code, words in cilinDatas.items():
        if word in words:
            wordEncodingDatas.append(code)
    return wordEncodingDatas



def wordSimilarity(word1,word2):
    """
    计算词语相似度,该方法是基于哈工大社会计算与信息检索研究中心同义词词林扩展版计算两个词的相似度
    :param word1:
    :param word2:
    :return:
    """
    word1Encode=wordEncoding(word1)
    word2Encode=wordEncoding(word2)
    if len(word1Encode)==0 or len(word2Encode)==0:
        return 0

    length=0
    depth=0
    for code1 in word1Encode:
        for code2 in word2Encode:
            if code1.find(code2)==0:
                length=0
                depth=len(code1)
            else:
                pass






def wordOrderSimilarity(order1,order2):
    """
    计算语句中词序的相似度,该值对语句对长度很敏感，句子长度越长，该值会越大；
    :param order1:
    :param order2:
    :return:
    """
    assert len(order1)==len(order2),u"两个句子的词序长度不相等"
    order1=np.array(order1)
    order2=np.array(order2)
    #词序向量差的二范数
    orderSubtract=np.linalg.norm(order1-order2,ord=2)
    #词序向量和二范数
    orderSum = np.linalg.norm(order1 + order2, ord=2)
    Sr=1-np.float32(orderSubtract)/np.float32(orderSum)
    return Sr


if __name__ == '__main__':
    order1= [1,2,3,4,5,6,7,8,9]
    order2= [1,2,3,9,5,6,7,8,4]
    # sr=wordOrderSimilarity(order1,order2)
    a={"1": '机器学习是一门多领域交叉学科', "2": '涉及概率论、统计学、逼近论、凸分析、算法复杂度理论等多门学科'}
    # semanticSimilarity(a['1'],a['2'])
    cilin()