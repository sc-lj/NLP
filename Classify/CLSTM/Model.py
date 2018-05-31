# coding:utf-8

import os,sys
sys.path.append(os.path.dirname(__file__))

import tensorflow as tf
from Config import *

class CNN():
    def __init__(self):
        self.arg=argument()
        # 要embedding的input必须是int类型
        self.input=tf.placeholder(shape=[None,self.arg.cnn_maxlen],name='input',dtype=tf.int32)
        self.label=tf.placeholder(shape=[None,],name='label',dtype=tf.float32)

    def embedding_variable(self):

        embed=tf.Variable(tf.random_uniform(shape=[1000,self.arg.cnn_embedding],minval=-0.25,maxval=0.25,dtype=tf.float32),name='embedding')
        return tf.nn.embedding_lookup(embed,self.input)

    def weight_variable(self,shape):
        weight=tf.Variable(tf.truncated_normal(shape=shape,mean=0,stddev=0.5,dtype=tf.float32),name='weight')
        # 将网络中所有层中的权重，依次通过tf.add_to_collectio加入到tf.GraphKeys.WEIGHTS中；
        tf.add_to_collection(tf.GraphKeys.WEIGHTS,weight)
        return weight

    def bias_variable(self,shape):
        bias=tf.Variable(tf.constant(0.,dtype=tf.float32,shape=shape),name='bias')
        return bias

    def create_model(self):
        convds=[]
        with tf.name_scope('conv'):
            for filt_size in self.arg.cnn_filter_size:
                # self.embedding_variable()维度是[batch,in_height,in_width]，而我们需要将其转化为[batch,in_height,in_width,in_channels]的形状
                input_embed=tf.expand_dims(self.embedding_variable(),-1)
                # filter的shape为[filter_height,filter_width,in_channels,out_channels]
                filter_shape = [filt_size, self.arg.cnn_embedding, 1, self.arg.cnn_filter_num]
                convd=tf.nn.conv2d(input_embed,filter=self.weight_variable(filter_shape),strides=(1,1,1,1),padding='VALID')
                bias_size=[self.arg.cnn_filter_num]
                convd=tf.nn.relu(tf.nn.bias_add(convd,self.bias_variable(bias_size)),name='bias')
                convds.append(convd)
                print(convd.get_shape().as_list())

        regularizer = tf.contrib.layers.l2_regularizer(0.1)
        self.reg=tf.contrib.layers.apply_regularization(regularizer)


        # self.new_convd=tf.concat(convds,3)
        # print(self.new_convd.get_shape().as_list())



if __name__ == '__main__':
    cnn=CNN()
    cnn.create_model()



