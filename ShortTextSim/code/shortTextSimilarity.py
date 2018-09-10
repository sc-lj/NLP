# coding:utf-8

"""
基于论文A Method for Measuring Sentence Similarity and its Application to Conversational Agents中的方法计算短文本的相似度
"""

import os,re,jieba,json
import numpy as np
from TextProcess import *

def readcorpus():
    f=open('./titlecorpus.json','r')
    data=f.read()
    f.close()
    data=json.loads(data)
    return data

corpus=readcorpus()

def wordPro(word):
    if "allword" in corpus:
        allNum = corpus['allword']
    else:allNum=sum(corpus.values())
    if word in corpus:
        num=corpus[word]
        wordpro=float(num)/float(allNum)
    else:
        wordpro=0
    return wordpro

def genSemanticSim(sentence,sentenceSet,semanticSimThreshold=0.2):
    """
    产生句子的语义向量
    :param sentence: 其中一个句子的词集
    :param sentenceSet: 要比较的两个句子的组成的词集
    :param semanticSimThreshold: 产生词相似度的阈值
    :return:
    """
    sentencevector=[]
    for i, word in enumerate(sentenceSet):
        wordpro = wordPro(word)
        if word in sentence:
            sentencevector.append(wordpro)
        else:
            maxSemanticSim1 = 0
            maxSemanticWord1 = 0
            for wor in sentence:
                wordSims = wordSimilarity(word, wor)
                if wordSims >= semanticSimThreshold and wordSims >= maxSemanticSim1:
                    maxSemanticSim1 = wordSims
                    maxSemanticWord1 = wor
            worpro = wordPro(maxSemanticWord1)
            maxSemanticSim1 = maxSemanticSim1 * wordpro * worpro
            sentencevector.append(maxSemanticSim1)
    return sentencevector


def semanticSimilarity(sentence1,sentence2,sentenceSet):
    """
    计算语句的语义相似度
    :param sentence1:
    :param sentence2:
    :param sentenceSet:
    :return:
    """
    semanticSimThreshold = 0.2
    sentence1vector=genSemanticSim(sentence1,sentenceSet,semanticSimThreshold)
    sentence2vector=genSemanticSim(sentence2,sentenceSet,semanticSimThreshold)

    Ss=np.array(sentence1vector)*np.array(sentence2vector)/(np.linalg.norm(sentence1vector,2)*np.linalg.norm(sentence2vector,2))
    return Ss

# 返回格式{code: '文字字符串', }
def cilin():
    print('------当前进行《同义词词林》的读入操作------')
    cilinFilePath = '../../Dataset/synonym/synonym_ex.txt'
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
    alpha=0.2
    beta=0.45
    word1Encode=wordEncoding(word1)
    word2Encode=wordEncoding(word2)
    if len(word1Encode)==0 or len(word2Encode)==0:
        return 0

    length=[]
    depth=[]
    for code1 in word1Encode:
        for code2 in word2Encode:
            if code1.find(code2)==0:
                length.append(0)
                depth.append(len(code1))
            else:
                dep=cmp(code1, code2)
                depth.append(dep)
                length.append(2*(len(code1)-dep))

    # 取最短的长度和深度
    depth=sorted(depth)[0]
    length=sorted(length)[-1]
    expbeta=lambda x:np.exp(beta*x*depth)
    Sw=np.exp(-alpha*length)*((expbeta(1)-expbeta(-1))/(expbeta(1)+expbeta(-1)))
    return Sw


def cmp(code1,code2):
    """
    比较两个同义词林代码不一样开始的位置。
    :param code1:
    :param code2:
    :return:
    """
    assert len(code1) == len(code2), u"两个词的同义词词林代码长度不一"
    for i in range(0,len(code1)):
        if code2[i]!=code1[i]:
            return i

def genWordOrder(sentence,sentenceSet,wordOrderThrehold=0.4):
    """
    计算词序向量
    :param sentence:
    :param sentenceSet:
    :param wordOrderThrehold:
    :return:
    """
    order = []
    for index, word in enumerate(sentenceSet):
        if word in sentence:
            order.append(sentence.index(word) + 1)
        else:
            maxWordOrder2 = 0
            maxWordOrderIndex2 = 0
            for ind, wor in enumerate(sentence):
                wordOrderSim = wordSimilarity(word, wor)
                if wordOrderSim >= wordOrderThrehold and wordOrderSim >= maxWordOrder2:
                    maxWordOrder2 = wordOrderSim
                    maxWordOrderIndex2 = ind
            order.append(maxWordOrderIndex2)
    return order

def wordOrderSimilarity(sentence1,sentence2,sentenceSet):
    """
    计算语句中词序的相似度,该值对语句对长度很敏感，句子长度越长，该值会越大；
    词序向量是在sentenceSet中的是否出现在sentence1或sentence2中，如果出现，就填写词在sentence1或2中的index。
    否则计算该词在sentence1或者2中最相似词的index。
    :param sentence1:
    :param sentence2:
    :param sentenceSet:
    :return:
    """
    wordOrderThrehold = 0.4
    order1=genWordOrder(sentence1,sentenceSet,wordOrderThrehold)
    order2=genWordOrder(sentence2,sentenceSet,wordOrderThrehold)

    assert len(order1)==len(order2),u"两个句子的词序长度不相等"
    order1=np.array(order1)
    order2=np.array(order2)
    #词序向量差的二范数
    orderSubtract=np.linalg.norm(order1-order2,ord=2)
    #词序向量和二范数
    orderSum = np.linalg.norm(order1 + order2, ord=2)
    Sr=1-np.float32(orderSubtract)/np.float32(orderSum)
    return Sr


def sentenceSimilarity(sentence1,sentence2):
    """
    计算句子的相似度
    :param sentence1:
    :param sentence2:
    :return:
    """
    sigma=0.85
    sentence1=cutWords(sentence1)
    sentence2=cutWords(sentence2)
    sentenceSet=list(set(sentence1).union(set(sentence2)))

    semanticSim=semanticSimilarity(sentence1,sentence2,sentenceSet)
    wordOrderSim=wordOrderSimilarity(sentence1,sentence2,sentenceSet)
    Sim=sigma*wordOrderSim+(1-sigma)*semanticSim
    return Sim




if __name__ == '__main__':
    order1= [1,2,3,4,5,6,7,8,9]
    order2= [1,2,3,9,5,6,7,8,4]
    # sr=wordOrderSimilarity(order1,order2)
    a={"1": '机器学习是一门多领域交叉学科', "2": '涉及概率论、统计学、逼近论、凸分析、算法复杂度理论等多门学科'}
    # semanticSimilarity(a['1'],a['2'])
    # cilin()
    # b=cmp("Ae07B07#","Ae07B07=")
    c=wordSimilarity("饲养户","小商贩")
    print(c)

