# coding:utf-8


import tensorflow as tf
from .config import *

class TextCNN():
    def __init__(self, vector_length,lable_length):
        """
        :param vector_length: 词向量的长度
        :param lable_length: 类别长度
        """
        self.filter_size=CNN.filter_size
        self.vector_length=vector_length
        self.lable_length=lable_length
        self.input_x=tf.placeholder(shape=[None,self.vector_length],name='input_x',dtype=tf.float32)
        self.input_y=tf.placeholder(shape=[None,self.lable_length],name='input_y',dtype=tf.float32)
    
    
    def weight(self,shape):
        """
        权重向量或者卷积向量
        :param shape:
        :return:
        """
        weight=tf.Variable(initial_value=tf.random_normal(shape,stddev=2.0),name='weight')
        return weight
    
    def biases(self,shape):
        biases=tf.Variable(initial_value=tf.random_normal(shape=shape,stddev=1.0),name='biases')
        return biases
    
    def create_model(self):
        with tf.name_scope('conv'):
            # 由于conv2d卷积shape为[height, width, in_channels, out_channels]，
            # in_channels: 图片的深度；在文本处理中深度为1，需要添加的一个1，增加其维度。
            filtersize=[self.filter_size,self.vector_length,1,CNN.filter_nums]
            convd=tf.nn.conv2d(self.input_x,filter=self.weight(filtersize),strides=(1,1,1,1),padding='VALID',name='convd')

        with tf.name_scope('pool'):
            max_pool=tf.nn.max_pool(convd,ksize=())
