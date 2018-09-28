# coding:utf-8

import tensorflow as tf
from collections import namedtuple

HParams=namedtuple("HParams","batch_size enc_timesteps dec_timesteps emb_dim con_layers kernel_size topic_size")

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

        self.topic=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size,hps.topic_size],name='topic')
        self.topic_position=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size,hps.enc_timesteps],name="topic_position")


    def ConvS2S(self):
        hps=self._hps
        vsize=self._vsize
        tsize=self._tsize

        with tf.variable_scope('convs2s'):
            targets=tf.unstack(tf.transpose(self.target))
            loss_weight=tf.unstack(tf.transpose(self.loss_weight))

            with tf.variable_scope("embedding"):
                word_emb=tf.get_variable(name='word_emb',shape=[vsize,hps.emb_dim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
                pos_emb=tf.get_variable(name='position_emb',shape=[hps.enc_timesteps,hps.emb_dim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
                topic_emb=tf.get_variable(name='topic_emb',shape=[tsize,hps.emb_dim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(0,stddev=0.1))

                # encoder端的embedding
                emb_encoder_inputs=tf.nn.embedding_lookup(word_emb,self.article)
                emb_encoder_positions=tf.nn.embedding_lookup(pos_emb,self.art_position)

                emb_encoder=tf.reduce_sum([emb_encoder_inputs,emb_encoder_positions],axis=0)

                # decoder端的embedding
                emb_decoder_inputs=tf.nn.embedding_lookup(word_emb,self.abstract)
                emb_decoder_positions=tf.nn.embedding_lookup(pos_emb,self.abs_position)

                self.emb_decoder=tf.reduce_sum([emb_decoder_inputs,emb_decoder_positions],axis=0)

                # 主题词的embedding
                emb_topic_inputs=tf.nn.embedding_lookup(topic_emb,self.topic)
                emb_topic_position=tf.nn.embedding_lookup(pos_emb,self.topic_position)

                emb_topic=tf.reduce_sum([emb_topic_inputs,emb_topic_position],axis=0)

            filters=[hps.kernel_size,hps.emb_dim,1,2*hps.emb_dim]
            padsize=int(hps.kernel_size/2)
            PAD_emb=tf.nn.embedding_lookup(word_emb,[1])


            # encoder端的卷积
            encoder_shape = emb_encoder.shape.as_list()
            self.encoder_outputs=[emb_encoder]
            for enc_layer in range(hps.con_layers):
                with tf.variable_scope("encoder_%d"%enc_layer):
                    emb_encoder = tf.reshape(emb_encoder, shape=[encoder_shape[0], encoder_shape[1], encoder_shape[2],1])  # batch_size,seq_len,emb_dim,1
                    filter=tf.Variable(initial_value=tf.truncated_normal(filters,stddev=0.01),name="filter")
                    emb_encoder=tf.nn.conv2d(emb_encoder,filter=filter,strides=[1,1,1,1],padding="VALID")
                    emb_encoder=tf.reshape(emb_encoder,shape=[hps.batch_size,-1,hps.emb_dim*2])

                    A,B=tf.split(emb_encoder,2,axis=2)
                    attn=tf.multiply(A,tf.nn.softmax(B))

                    # 对attn进行padding，使用PAD的emb进行padding，但是文献中建议使用0向量padding。
                    attn=tf.pad(attn,paddings=[[0,0],[padsize,padsize],[0,0]],constant_values=PAD_emb)
                    emb_encoder=attn+self.encoder_outputs[-1]
                    self.encoder_outputs.append(emb_encoder)


            # topic 的卷积
            topic_shape = emb_topic.shape.as_list()
            self.topic_outputs=[emb_topic]
            for topic_layer in range(hps.con_layers):
                with tf.variable_scope("topic_%d"%topic_layer):
                    emb_topic = tf.reshape(emb_topic, shape=[topic_shape[0], topic_shape[1], topic_shape[2],1])  # batch_size,seq_len,emb_dim,1
                    filter=tf.Variable(initial_value=tf.truncated_normal(filters,stddev=0.01),name="filter")
                    emb_topic=tf.nn.conv2d(emb_topic,filter=filter,strides=[1,1,1,1],padding='VALID')
                    emb_topic=tf.reshape(emb_topic,shape=[hps.batch_size,-1,hps.emb_dim*2])
                    A,B=tf.split(emb_topic,2,axis=2)
                    attn=tf.multiply(A,tf.nn.softmax(B))

                    # 对attn进行padding，使用PAD的emb进行padding，但是文献中建议使用0向量padding。
                    attn=tf.pad(attn,paddings=[[0,0],[padsize,padsize],[0,0]],constant_values=PAD_emb)
                    emb_topic=attn+self.topic_outputs[-1]
                    self.topic_outputs.append(emb_topic)


    def decoder(self):
        hps=self._hps
        emb_decoder = [tf.reshape(emb, shape=[hps.batch_size, 1, hps.emb_dim, 1]) for emb in self.emb_decoder]
        emb_decoder = tf.concat(emb_decoder, axis=1)  # batch_size,seq_len,emb_dim,1
        for dec_layer in range(hps.con_layers):
            with tf.variable_scope("decoder_%d" % dec_layer):
                pass






