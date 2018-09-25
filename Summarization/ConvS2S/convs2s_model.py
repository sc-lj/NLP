# coding:utf-8

import tensorflow as tf
from collections import namedtuple

HParams=namedtuple("HParams","batch_size enc_timesteps dec_timesteps emb_dim con_layers kernel_size")

class ConvS2SModel():
    def __init__(self,vsize,tsize,hps):
        self._hps=hps
        self._vsize=vsize
        self._tsize=tsize

    def _add_placeholder(self):
        hps=self._hps
        self.article=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size,hps.enc_timesteps],name="articles")
        self.art_position=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size,hps.enc_timesteps],name='article_position')

        self.abstract=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size,hps.dec_timesteps],name='abstracts')
        self.abs_position=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size,hps.dec_timesteps],name='abstract_position')

        self.target=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size,hps,hps.dec_timesteps],name='targets')
        self.loss_weight=tf.placeholder(dtype=tf.float32,shape=[hps.batch_size,hps.dec_timesteps],name='loss_weight')

        self.article_len=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size],name='article_len')
        self.abstract_len=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size],name='abstract_len')


    def con(self,input_x):
        hps=self._hps
        weights = tf.get_variable("weight",shape=[hps.emb_dim*hps.kernel_size,2*hps.emb_dim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(0,stddev=1e-4))
        bias=tf.get_variable('bias',shape=[2*hps.emb_dim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1e-4))
        convs=tf.nn.xw_plus_b(input_x,weights,bias,name="convolution")
        convs=tf.reshape(convs,shape=[2,hps.emb_dim])
        convsA=convs[1,:]
        convsB=convs[2,:]
        glu=tf.multiply(convsA,tf.nn.sigmoid(convsB),name="glu")
        return glu


    def ConvS2S(self):
        hps=self._hps
        vsize=self._vsize
        tsize=self._tsize

        with tf.variable_scope('convs2s'):
            encoder_inputs=tf.unstack(tf.transpose(self.article))
            encoder_positions=tf.unstack(tf.transpose(self.art_position))

            decoder_inputs=tf.unstack(tf.transpose(self.abstract))
            decoder_positions=tf.unstack(tf.transpose(self.abs_position))

            targets=tf.unstack(tf.transpose(self.target))
            loss_weight=tf.unstack(tf.transpose(self.loss_weight))


            with tf.variable_scope("embedding"):
                word_emb=tf.get_variable(name='word_emb',shape=[vsize,hps.emb_dim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
                pos_emb=tf.get_variable(name='position_emb',shape=[hps.enc_timesteps,hps.emb_dim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
                topic_emb=tf.get_variable(name='topic_emb',shape=[tsize,hps.emb_dim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(0,stddev=0.1))

                emb_encoder_inputs=[tf.nn.embedding_lookup(word_emb,encoder_input) for encoder_input in encoder_inputs]
                emb_ebcoder_positions=[tf.nn.embedding_lookup(pos_emb,encoder_position) for encoder_position in encoder_positions]

                emb_decoder_inputs=[tf.nn.embedding_lookup(topic_emb,decoder_input) for decoder_input in decoder_inputs]
                emb_decoder_positions=[tf.nn.embedding_lookup(pos_emb,decoder_position) for decoder_position in decoder_positions]


            for enc_layer in range(hps.con_layers):
                with tf.variable_scope("encoder_%d"%enc_layer):


                    pass






