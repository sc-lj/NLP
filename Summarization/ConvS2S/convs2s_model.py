# coding:utf-8

import tensorflow as tf
from collections import namedtuple

HParams=namedtuple("HParams","batch_size enc_timesteps dec_timesteps emb_dim con_layers")

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
        self.article_len=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size],name='article_len')
        self.abstract_len=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size],name='abstract_len')
        self.loss_weight=tf.placeholder(dtype=tf.float32,shape=[hps.batch_size,hps.dec_timesteps],name='loss_weight')


    def Conv(self):
        hps=self._hps
        vsize=self._vsize
        tsize=self._tsize

        with tf.variable_scope('convs2s'):
            encoder_inputs=tf.unstack(tf.transpose(self.article))
            decoder_inputs=tf.unstack(tf.transpose(self.abstract))
            targets=tf.unstack(tf.transpose(self.target))
            loss_weight=tf.unstack(tf.transpose(self.loss_weight))

            with tf.variable_scope("embedding"):
                word_emb=tf.get_variable(name='word_emb',shape=[vsize,hps.emb_dim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
                pos_emb=tf.get_variable(name='position_emb',shape=[hps.enc_timesteps,hps.emb_dim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
                topic_emb=tf.get_variable(name='topic_emb',shape=[tsize,hps.emb_dim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(0,stddev=0.1))




