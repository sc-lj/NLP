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
                emb_encoder_positions=[tf.nn.embedding_lookup(pos_emb,encoder_position) for encoder_position in encoder_positions]

                emb_encoder=[]
                for i in range(len(emb_encoder_inputs)):
                    emb_encoder.append(tf.reduce_sum([emb_encoder_inputs[i],emb_encoder_positions[i]],axis=0))

                emb_decoder_inputs=[tf.nn.embedding_lookup(topic_emb,decoder_input) for decoder_input in decoder_inputs]
                emb_decoder_positions=[tf.nn.embedding_lookup(pos_emb,decoder_position) for decoder_position in decoder_positions]

                emb_decoder=[]
                for i in range(len(emb_decoder_inputs)):
                    emb_decoder.append(tf.reduce_sum([emb_decoder_inputs[i],emb_decoder_positions[i]],axis=0))

            emb_encoder=[tf.reshape(emb,shape=[hps.batch_size,1,hps.emb_dim,1]) for emb in emb_encoder]
            emb_encoder=tf.concat(emb_encoder,axis=1)# batch_size,seq_len,emb_dim,1

            filters=[hps.kernel_size,hps.emb_dim,1,2*hps.emb_dim]
            for enc_layer in range(hps.con_layers):
                with tf.variable_scope("encoder_%d"%enc_layer):
                    emb_encoder=tf.nn.conv2d(emb_encoder,filter=filters,strides=[1,1,1,1],padding="VALID")
                    emb_encoder=tf.reshape(emb_encoder,shape=[hps.batch_size,-1,hps.emb_dim*2])












