# coding:utf-8


import tensorflow as tf


class TextCNN():
    def __init__(self, vertor_length,lable_length):
        self.input_x=tf.placeholder(shape=[None,vertor_length],name='input_x',dtype=tf.int32)
        self.input_y=tf.placeholder(shape=[None,lable_length],name='input_y',dtype=tf.int32)
    
    
    def weight(self):
        weight=tf.Variable(validate_shape=[],initial_value=tf.random_normal_initializer())
        
    
    
