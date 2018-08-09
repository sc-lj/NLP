# coding:utf-8

"""
基于论文A Method for Measuring Sentence Similarity and its Application to Conversational Agents中的方法计算短文本的相似度
"""

import os,re,jieba
import numpy as np

def stopwords():
    """
    读取停用词
    :return:
    """
    with open('../../Dataset/stopwords/stopwords.txt','r') as f:
        stopword=f.read()
    return stopword

def cutWords(content):
    """
    分句子后，然后分词
    :param content:
    :return:
    """
    content=content.lower()
    stopword = stopwords()
    splitdot=re.compile('[。，？！：,.、;]')# 断句的标点呼号
    sentences=splitdot.split(content)# 切分成多个语句
    words=[]
    for sentence in sentences:
        rule = re.compile(r"[^a-zA-Z\u4e00-\u9fa5]")  # 只留字母、中文。
        sentence=rule.sub('',sentence)
        word=jieba.cut(sentence)#结巴对句子进行分词
        words.extend(list(word))

    # 过滤停用词
    filterstopword=list(filter(lambda x:x if x not in stopword else None, words))
    return filterstopword


def sentenceSimilarity(sentence1,sentence2):
    """
    计算语句相似度
    :param sentence1:
    :param sentence2:
    :return:
    """



def wordSimilarity(word1,word2):
    """
    计算词语相似度,该方法是基于
    :param word1:
    :param word2:
    :return:
    """


def wordOrderSimilarity(order1,order2):
    """
    计算语句中词序的相似度,该值对语句对长度很敏感，句子长度越长，该值会越大；
    :param order1:
    :param order2:
    :return:
    """
    order1=np.array(order1)
    order2=np.array(order2)
    #词序向量差的二范数
    orderSubtract=np.linalg.norm(order1-order2,ord=2)
    #词序向量和二范数
    orderSum = np.linalg.norm(order1 + order2, ord=2)
    Sr=1-np.float32(orderSubtract)/np.float32(orderSum)
    # Sr=Sr/len(order1)
    return Sr


if __name__ == '__main__':
    order1= [1,2,3,4,5,6,7,8,9]
    order2= [1,2,3,9,5,6,7,8,4]
    sr=wordOrderSimilarity(order1,order2)
    print(sr)



