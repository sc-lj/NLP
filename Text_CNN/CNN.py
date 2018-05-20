# coding:utf-8


import tensorflow as tf
from .config import *
import os

class TextCNN():
    def __init__(self, vector_length,lable_length):
        """
        :param vector_length: 词向量的长度
        :param lable_length: 类别长度
        """
        self.filter_sizes=seq.filter_sizes
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
            pools=[]
            for i,filter_size in enumerate(self.filter_sizes):
                # 由于conv2d卷积shape为[height, width, in_channels, out_channels]，
                # in_channels: 图片的深度；在文本处理中深度为1，需要添加的一个1，增加其维度。
                filtersize=[filter_size,self.vector_length,1,seq.filter_nums]
                # 'SAME'模式下的convd的形状是：[1,sequence_length-filter_size+1,1,1]
                convd=tf.nn.conv2d(self.input_x,filter=self.weight(filtersize),strides=(1,1,1,1),padding='SAME',name='convd')
                biase_shape=[seq.filter_nums]
                convd=tf.nn.relu(tf.nn.bias_add(convd,self.biases(biase_shape)),name='relu')
                # convd的shape为：[batch_size,1]
                pooled=tf.nn.max_pool(convd,ksize=(1,seq.seqence_length-filter_size+1,1,1),strides=(1,1,1,1),padding='SAME',name='pool')
                pools.append(pooled)

            num_filters_total=seq.filter_nums*len(self.filter_sizes)
            self.h_pool=tf.concat(pools,3)
            # pool_flat的shape为：[batch_size,num_filters_total]
            self.pool_flat=tf.reshape(self.h_pool,[-1,num_filters_total])

        # dropout layer随机地选择一些神经元
        with tf.name_scope('dropout'):
            self.h_drop=tf.nn.dropout(self.pool_flat,seq.dropout_prob)

        with tf.name_scope('output'):
            W=self.weight([num_filters_total,self.lable_length])
            b=self.biases([self.lable_length])
            self.score=tf.nn.xw_plus_b(self.h_drop,W,b,name='score')
            self.prediction=tf.argmax(self.score,1,name='prediction')

        with tf.name_scope('loss'):
            losses=tf.nn.softmax_cross_entropy_with_logits(self.score,self.input_y)
            self.loss=tf.reduce_mean(losses)

        with tf.name_scope('accuracy'):
            correct_predictions=tf.equal(self.prediction,tf.argmax(self.input_y,1))
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions,'float'),name='accuracy')

    def train_model(self,out_dir):
        #
        global_step=tf.Variable(0,name='global_step',trainable=False)

        # 定义优化算法
        optimizer=tf.train.AdamOptimizer(1e-4)

        grads_and_vars=optimizer.compute_gradients(self.loss)

        # 在参数上进行梯度更新,每执行一次 train_op 就是一次训练步骤
        train_op=optimizer.apply_gradients(grads_and_vars,global_step)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir,'checkpoints'))

        checkpoint_prefix = os.path.join(checkpoint_dir,'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver=tf.train.Saver(tf.all_variables())






