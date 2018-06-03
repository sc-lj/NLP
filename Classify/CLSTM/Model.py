# coding:utf-8

import os,sys
sys.path.append(os.path.dirname(__file__))

import tensorflow as tf
from Config import *

arg=argument()
class CNN():
    def __init__(self):
        self.arg=arg
        # 要embedding的input必须是int类型
        self.input=tf.placeholder(shape=[None,self.arg.cnn_maxlen],name='input',dtype=tf.int32)

        self.cnn_model()

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

    def cnn_model(self):
        convds=[]
        with tf.name_scope('conv'):
            for i,filt_size in enumerate(self.arg.cnn_filter_size):

                # self.embedding_variable()维度是[batch,in_height,in_width]，而我们需要将其转化为[batch,in_height,in_width,in_channels]的形状
                input_embed=tf.expand_dims(self.embedding_variable(),-1)
                # filter的shape为[filter_height,filter_width,in_channels,out_channels]
                filter_shape = [filt_size, self.arg.cnn_embedding, 1, self.arg.cnn_filter_num]
                convd=tf.nn.conv2d(input_embed,filter=self.weight_variable(filter_shape),strides=(1,1,1,1),padding='VALID')
                bias_size=[self.arg.cnn_filter_num]
                convd=tf.nn.relu(tf.nn.bias_add(convd,self.bias_variable(bias_size)),name='bias')
                convds.append(convd)

        self.new_convds=[]
        max_demen=max([convd.get_shape()[1] for convd in convds])
        for convd in convds:
            shape=convd.get_shape()[1]
            if shape<max_demen:
                pad_num=int(max_demen-shape)
                # 将convd shape [None,297,1,128]==>[None,297+pad_num,1,128]
                convd = tf.pad(convd,paddings=[[0,0],[0,pad_num],[0,0],[0,0]])
            # [None,297+pad_num,1,128]==>[None,297+pad_num,128]
            conv = tf.reshape(convd, [-1, int(max_demen), self.arg.cnn_filter_num])
            self.new_convds.append(conv)

        self.new_convds=tf.concat(self.new_convds,2)
        regularizer = tf.contrib.layers.l2_regularizer(0.1)
        self.reg=tf.contrib.layers.apply_regularization(regularizer)

class LSTM(CNN):
    def __init__(self):
        CNN.__init__(self)
        self.label = tf.placeholder(shape=[None, ], name='label', dtype=tf.float32)

        self.bilstm()

    def rnn_cell(self):
        # BasicLSTMCell类没有实现clipping，projection layer，peep-hole等一些lstm的高级变种，仅作为一个基本的basicline结构存在。
        # tf.nn.rnn_cell.BasicLSTMCell
        # LSTMCell类实现了clipping，projection layer，peep-hole。
        # tf.nn.rnn_cell.LSTMCell
        return tf.nn.rnn_cell.LSTMCell(self.arg.rnn_hidden_unite)

    def gru_cell(self):
        return tf.nn.rnn_cell.GRUCell(self.arg.rnn_hidden_unite)

    def bilstm(self):
        if self.arg.is_lstm:
            cell=self.rnn_cell()
        else:
            cell=self.gru_cell()
        # dropout
        cell=tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=self.arg.lstm_dropout)
        # 多层rnn
        # cell=tf.nn.rnn_cell.MultiRNNCell([cell]*self.arg.lstm_layer_num)

        initial_state=cell.zero_state(self.arg.batch_size,dtype=tf.float32)

        inputs=self.new_convds
        sequence_length=tf.shape(inputs)[1]
        sequence_length = tf.expand_dims(sequence_length, axis=0, name='sequence_length')

        if self.arg.is_bidirectional:
            outputs,states=tf.nn.bidirectional_dynamic_rnn(cell,cell,inputs=inputs,initial_state_bw=initial_state,initial_state_fw=initial_state)
        else:
            outputs,states=tf.nn.dynamic_rnn(cell,inputs=inputs,initial_state=initial_state,dtype=tf.float32,)





if __name__ == '__main__':
    rnn=LSTM()



