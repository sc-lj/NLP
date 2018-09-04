# coding:utf-8

import re,os
import tensorflow as tf


class DPCWS():
    def __init__(self,files):
        self.lr=0.02# 学习率
        self.window=5 #窗口大小
        self.CFDimension=50 # 字符特征的维度
        self.unit=300 # 隐藏层大小

        self.files=files
        self.X=tf.placeholder(shape=[None,200],name="X",dtype=tf.float32)
        self.Y=tf.placeholder(shape=[None,200],name="Y",dtype=tf.int32)

    def vocab(self):
        f=open(self.files)
        lines=f.readlines()
        f.close()
        voca=set()
        for line in lines:
            if len(line.strip())==0:
                continue
            word,mark=line.split("\t")
            voca.add(word)
        print(voca)

    def weight(self,shape):
        return tf.Variable(tf.truncated_normal(shape,0,0.2),name="weight",dtype=tf.float32)

    def biase(self,shape):
        return tf.Variable(tf.constant(0,dtype=tf.float32,shape=shape),name="biase")

    def model(self):
        pass


if __name__ == '__main__':
    dpcws=DPCWS("./pku_corpus.txt")
    dpcws.vocab()