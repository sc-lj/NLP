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
                self.word_emb=tf.get_variable(name='word_emb',shape=[vsize,hps.emb_dim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
                self.pos_emb=tf.get_variable(name='position_emb',shape=[hps.enc_timesteps,hps.emb_dim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
                topic_emb=tf.get_variable(name='topic_emb',shape=[tsize,hps.emb_dim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(0,stddev=0.1))

                # encoder端的embedding
                emb_encoder_inputs=tf.nn.embedding_lookup(self.word_emb,self.article)
                emb_encoder_positions=tf.nn.embedding_lookup(self.pos_emb,self.art_position)

                emb_encoder=tf.reduce_sum([emb_encoder_inputs,emb_encoder_positions],axis=0)

                # 主题词的embedding
                emb_topic_inputs=tf.nn.embedding_lookup(topic_emb,self.topic)
                emb_topic_position=tf.nn.embedding_lookup(self.pos_emb,self.topic_position)

                emb_topic=tf.reduce_sum([emb_topic_inputs,emb_topic_position],axis=0)

            self.filters=[hps.kernel_size,hps.emb_dim,1,2*hps.emb_dim]
            padsize=int(hps.kernel_size/2)
            self.PAD_emb=tf.nn.embedding_lookup(self.word_emb,[1])


            # encoder端的卷积
            last_encoder_outputs=emb_encoder
            for enc_layer in range(hps.con_layers):
                with tf.variable_scope("encoder_%d"%enc_layer):
                    # 对attn进行padding，使用PAD的emb进行padding，但是文献中建议使用0向量padding。
                    emb_encoder = tf.pad(emb_encoder, paddings=[[0, 0], [padsize, padsize], [0, 0]], constant_values=self.PAD_emb)
                    encoder_shape = emb_encoder.shape.as_list()
                    emb_encoder = tf.reshape(emb_encoder, shape=[encoder_shape[0], encoder_shape[1], encoder_shape[2],1])  # batch_size,seq_len,emb_dim,1
                    filter=tf.Variable(initial_value=tf.truncated_normal(self.filters,stddev=0.01),name="filter")
                    emb_encoder=tf.nn.conv2d(emb_encoder,filter=filter,strides=[1,1,1,1],padding="VALID")
                    emb_encoder=tf.reshape(emb_encoder,shape=[hps.batch_size,-1,hps.emb_dim*2])

                    A,B=tf.split(emb_encoder,2,axis=2)
                    attn=tf.multiply(A,tf.nn.softmax(B))

                    emb_encoder=attn+last_encoder_outputs
                    last_encoder_outputs=emb_encoder

            self.encoder_outputs=emb_encoder

            # topic 的卷积
            last_topic_outputs=emb_topic
            for topic_layer in range(hps.con_layers):
                with tf.variable_scope("topic_%d"%topic_layer):
                    # 对attn进行padding，使用PAD的emb进行padding，但是文献中建议使用0向量padding。
                    emb_topic = tf.pad(emb_topic, paddings=[[0, 0], [padsize, padsize], [0, 0]], constant_values=self.PAD_emb)
                    topic_shape = emb_topic.shape.as_list()
                    emb_topic = tf.reshape(emb_topic, shape=[topic_shape[0], topic_shape[1], topic_shape[2],1])  # batch_size,seq_len,emb_dim,1
                    filter=tf.Variable(initial_value=tf.truncated_normal(self.filters,stddev=0.01),name="filter")
                    emb_topic=tf.nn.conv2d(emb_topic,filter=filter,strides=[1,1,1,1],padding='VALID')
                    emb_topic=tf.reshape(emb_topic,shape=[hps.batch_size,-1,hps.emb_dim*2])
                    A,B=tf.split(emb_topic,2,axis=2)
                    attn=tf.multiply(A,tf.nn.softmax(B))

                    emb_topic=attn+last_topic_outputs
                    last_topic_outputs=emb_topic

            self.topic_outputs=emb_topic

    def decoder(self):
        hps=self._hps
        padsize = int(hps.kernel_size / 2)
        # decoder端的embedding
        emb_decoder_inputs = tf.nn.embedding_lookup(self.word_emb, self.abstract)
        emb_decoder_positions = tf.nn.embedding_lookup(self.pos_emb, self.abs_position)

        _emb_decoder = tf.reduce_sum([emb_decoder_inputs, emb_decoder_positions], axis=0)
        last_decoder_outputs =emb_decoder= _emb_decoder
        for dec_layer in range(hps.con_layers):
            with tf.variable_scope("decoder_%d" % dec_layer):
                # 对attn进行padding，使用PAD的emb进行padding，但是文献中建议使用0向量padding。
                emb_decoder = tf.pad(emb_decoder, paddings=[[0, 0], [padsize, padsize], [0, 0]], constant_values=self.PAD_emb)
                decoder_shape = emb_decoder.shape.as_list()
                emb_decoder = tf.reshape(emb_decoder, shape=[decoder_shape[0], decoder_shape[1], decoder_shape[2],1])  # batch_size,seq_len,emb_dim,1
                filter = tf.Variable(initial_value=tf.truncated_normal(self.filters, stddev=0.01), name="filter")
                emb_decoder = tf.nn.conv2d(emb_decoder, filter=filter, strides=[1, 1, 1, 1], padding='VALID')
                emb_decoder = tf.reshape(emb_decoder, shape=[hps.batch_size, -1, hps.emb_dim * 2])
                A, B = tf.split(emb_decoder, 2, axis=2)
                attn = tf.multiply(A, tf.nn.softmax(B))

                emb_decoder = attn + last_decoder_outputs # batch,seq_len,dim
                emb_decoder=tf.reshape(emb_decoder,shape=[-1,hps.emb_dim]) # batch*seq_len,dim
                weight=tf.get_variable(name="attention_weight",shape=[hps.emb_dim,hps.emb_dim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01))
                bias=tf.get_variable(name="attention_bias",shape=[hps.emb_dim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.001))

                attn2=tf.nn.xw_plus_b(emb_decoder,weight,bias)
                attn2=tf.reshape(attn2,shape=[hps.batch_size,-1,hps.emb_dim])+_emb_decoder
                _attn_weight=tf.matmul(attn2,tf.transpose(self.encoder_outputs,[0,2,1])) #batch,decoder_seq_len,encoder_seq_len
                _attn_weight=tf.reshape(_attn_weight,shape=[-1,_attn_weight.shape[2]])
                attn_weight=tf.nn.softmax(_attn_weight,axis=1)
                attn_weight=tf.reshape(attn_weight,shape=[hps.batch_size,attn_weight.shape[1],-1])

                attns=tf.reduce_sum()







