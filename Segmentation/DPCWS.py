# coding:utf-8

import re,os
import tensorflow as tf


class DPCWS():
    def __init__(self):
        self.lr=0.02# 学习率
        self.window=5 #窗口大小
        self.CFDimension=50 # 字符特征的维度
        self.unit=300 # 隐藏层大小
        self.X=tf.placeholder(shape=[None,200],name="X",dtype=tf.float32)
        self.Y=tf.placeholder(shape=[None,])

    def weight(self,shape):
        return tf.Variable(tf.truncated_normal(shape,0,0.2),name="weight",dtype=tf.float32)

    def biase(self,shape):
        return tf.Variable(tf.constant(0,dtype=tf.float32,shape=shape),name="biase")

    def model(self):
        pass
