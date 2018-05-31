# coding:utf-8

import os,sys
sys.path.append(os.path.dirname(__file__))

import tensorflow as tf
from Config import *

class CNN():
    def __init__(self):
        self.arg=argument()
        self.input=tf.placeholder(shape=[None,self.arg.maxlen],name='input',dtype=tf.float32)
        self.label=tf.placeholder(shape=[None,],name='label',dtype=tf.float32)

    def embedding_variable(self):
        embed=tf.Variable(tf.random_uniform(shape=[],minval=-0.25,maxval=0.25),name='embedding')
        return tf.nn.embedding_lookup(embed,self.input)

    def weight_variable(self,shape):
        weight=tf.Variable(tf.truncated_normal(shape=shape,mean=0,stddev=0.5,name='weight'))
        return weight

    def bias_variable(self,shape):
        bias=tf.Variable(tf.constant(0,dtype=tf.float32,shape=shape),name='bias')
        return bias

    def create_model(self):

        with tf.name_scope('conv'):
            filter_size=()
            conv=tf.nn.conv2d(self.embedding_variable(),filter=())
        pass