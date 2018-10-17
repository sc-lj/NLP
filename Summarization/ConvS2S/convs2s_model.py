# coding:utf-8

import tensorflow as tf
from collections import namedtuple
import numpy as np
from Data import *

HParams=namedtuple("HParams","batch_size enc_timesteps dec_timesteps emb_dim con_layers kernel_size min_input_len")

class ConvS2SModel():
    def __init__(self,hps,vocab,num_gpus=0):
        self._hps=hps
        self._vsize=vocab.NumIds()#词汇量大小
        self._tsize=vocab.TopicIds()#主题词汇量大小
        self._vocab=vocab
        self._num_gpus = num_gpus
        self._add_placeholder()
        self._embedding()

    def _add_placeholder(self):
        hps=self._hps
        # 文本数据，及其位置、topic的输入
        self.article=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size,hps.enc_timesteps],name="articles")
        self.article_position=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size,hps.enc_timesteps],name='article_position')

        # 摘要数据，及其位置、topic的输入
        self.abstract=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size,hps.dec_timesteps],name='abstracts')
        self.abstract_position=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size,hps.dec_timesteps],name='abstracts_position')
        self.topic_to_vocab=tf.placeholder(dtype=tf.int32,shape=[self._tsize,2],name="abstractIsTopic")

        #
        self.target=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size,hps.dec_timesteps],name='targets')
        self.loss_weight=tf.placeholder(dtype=tf.float32,shape=[hps.batch_size,hps.dec_timesteps],name='loss_weight')

        # 文本和摘要的实际长度
        self.article_len=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size],name='article_len')
        self.abstract_len=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size],name='abstract_len')
        # 组成摘要的词是否在topic中
        self.indicator=tf.placeholder(dtype=tf.float32,shape=[self._vsize],name="indicator")

        self.sample_caption=tf.placeholder(dtype=tf.int32,shape=[hps.batch_size,hps.dec_timesteps],name='sample_caption')

    def _embedding(self):
        hps=self._hps
        vsize=self._vsize
        tsize=self._tsize

        with tf.variable_scope("embedding"):
            self.vocab_emb = tf.get_variable(name='word_emb', shape=[vsize, hps.emb_dim], dtype=tf.float32,
                                             initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
            self.pos_emb = tf.get_variable(name='position_emb', shape=[hps.enc_timesteps, hps.emb_dim],dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
            self.topic_emb = tf.get_variable(name='topic_emb', shape=[tsize, hps.emb_dim], dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer(0, stddev=0.1))

    def _next_device(self):
        """Round robin the gpu device. (Reserve last gpu for expensive op)."""
        if self._num_gpus == 0:
            return ''
        dev = '/gpu:%d' % self._cur_gpu
        if self._num_gpus > 1:
            self._cur_gpu = (self._cur_gpu + 1) % (self._num_gpus - 1)
        return dev

    def _get_gpu(self, gpu_id):
        if self._num_gpus <= 0 or gpu_id >= self._num_gpus:
            return ''
        return '/gpu:%d' % gpu_id

    def _Encoder(self):
        hps=self._hps

        with tf.variable_scope('convs2s'),tf.device(self._next_device()):
            # encoder端文章的embedding
            emb_encoder_inputs=tf.nn.embedding_lookup(self.vocab_emb,self.article)
            emb_encoder_positions=tf.nn.embedding_lookup(self.pos_emb,self.article_position)

            _emb_encoder=tf.reduce_sum([emb_encoder_inputs,emb_encoder_positions],axis=0)

            # encoder端文章主题词的embedding
            emb_topic_inputs=self.topic_embbeding(self.article)
            emb_topic_position=tf.nn.embedding_lookup(self.pos_emb,self.article_position)

            _emb_topic=tf.reduce_sum([emb_topic_inputs,emb_topic_position],axis=0)


            self.filters=[hps.kernel_size,hps.emb_dim,1,2*hps.emb_dim]
            padsize=int(hps.kernel_size/2)
            self.PAD_emb=tf.nn.embedding_lookup(self.vocab_emb,[1])

            with tf.variable_scope("text_encoder"):
                # encoder端的卷积
                last_encoder_outputs=emb_encoder=_emb_encoder
                for enc_layer in range(hps.con_layers):
                    with tf.variable_scope("encoder_%d"%enc_layer),tf.device(self._next_device()):
                        # 对attn进行padding，使用PAD的emb进行padding，但是文献中建议使用0向量padding。
                        emb_encoder = tf.pad(emb_encoder, paddings=[[0, 0], [padsize, padsize-1], [0, 0]], constant_values=self.PAD_emb)
                        emb_encoder = tf.expand_dims(emb_encoder,-1) # batch_size,seq_len,emb_dim,1
                        filter=tf.Variable(initial_value=tf.truncated_normal(self.filters,stddev=0.01),name="filter")
                        emb_encoder=tf.nn.conv2d(emb_encoder,filter=filter,strides=[1,1,1,1],padding="VALID")
                        emb_encoder=tf.reshape(emb_encoder,shape=[hps.batch_size,-1,hps.emb_dim*2])
                        A,B=tf.split(emb_encoder,2,axis=2)
                        attn=tf.multiply(A,tf.nn.softmax(B))

                        # residual connections
                        emb_encoder=attn+last_encoder_outputs
                        last_encoder_outputs=emb_encoder

                self.encoder_outputs=emb_encoder# batch,seq_len,dim
                self.encoder_out=emb_encoder+_emb_encoder

            with tf.variable_scope("topic_encoder"):
                # topic 的卷积
                last_topic_outputs=emb_topic=_emb_topic
                for topic_layer in range(hps.con_layers):
                    with tf.variable_scope("topic_%d"%topic_layer),tf.device(self._next_device()):
                        # 对attn进行padding，使用PAD的emb进行padding，但是文献中建议使用0向量padding。
                        emb_topic = tf.pad(emb_topic, paddings=[[0, 0], [padsize, padsize-1], [0, 0]], constant_values=self.PAD_emb)
                        emb_topic = tf.expand_dims(emb_topic,-1)  # batch_size,seq_len,emb_dim,1
                        # the convolution
                        filter=tf.Variable(initial_value=tf.truncated_normal(self.filters,stddev=0.01),name="filter")
                        emb_topic=tf.nn.conv2d(emb_topic,filter=filter,strides=[1,1,1,1],padding='VALID')
                        emb_topic=tf.reshape(emb_topic,shape=[hps.batch_size,-1,hps.emb_dim*2])
                        # the gate linear unit
                        A,B=tf.split(emb_topic,2,axis=2)
                        attn=tf.multiply(A,tf.nn.softmax(B))

                        # residual connections
                        emb_topic=attn+last_topic_outputs
                        last_topic_outputs=emb_topic

                self.topic_outputs=emb_topic
                self.topic_out=emb_topic+_emb_topic

    def _Decoder(self,abstract,abstract_position,reuse=False):
        """对文本摘要的decoder端的计算"""
        hps=self._hps
        padsize = int(hps.kernel_size / 2)
        with tf.variable_scope("text_decoder",reuse=reuse):
            # decoder端文本摘要的embedding
            emb_decoder_inputs = tf.nn.embedding_lookup(self.vocab_emb, abstract)
            emb_decoder_positions = tf.nn.embedding_lookup(self.pos_emb, abstract_position)

            _emb_decoder = tf.reduce_sum([emb_decoder_inputs, emb_decoder_positions], axis=0)
            last_decoder_outputs =emb_decoder= _emb_decoder# batch,seq_len_target,dim
            attn_c=[]# 用于加到topic端的h
            for dec_layer in range(hps.con_layers):
                with tf.variable_scope("decoder_%d" % dec_layer),tf.device(self._next_device()):
                    # 对attn进行padding，使用PAD的emb进行padding，但是文献中建议使用0向量padding。
                    emb_decoder = tf.pad(emb_decoder, paddings=[[0, 0], [padsize, padsize-1], [0, 0]], constant_values=self.PAD_emb)
                    emb_decoder = tf.expand_dims(emb_decoder,-1)  # batch_size,seq_len,emb_dim,1
                    # convolution
                    filter = tf.Variable(initial_value=tf.truncated_normal(self.filters, stddev=0.01), name="filter")
                    emb_decoder = tf.nn.conv2d(emb_decoder, filter=filter, strides=[1, 1, 1, 1], padding='VALID')
                    emb_decoder = tf.reshape(emb_decoder, shape=[hps.batch_size, -1, hps.emb_dim * 2])
                    # the gate linear unit
                    A, B = tf.split(emb_decoder, 2, axis=2)
                    attn_h = tf.multiply(A, tf.nn.softmax(B))
                    # residual connections
                    attn_h = attn_h + last_decoder_outputs # batch,seq_len_target,dim
                    # attention mechanism
                    emb_decoder=tf.reshape(attn_h,shape=[-1,hps.emb_dim]) # batch*seq_len,dim
                    attn_d=self.xw_plus_b(emb_decoder,shapes=[hps.emb_dim,hps.emb_dim])
                    attn_d=tf.reshape(attn_d,shape=[hps.batch_size,-1,hps.emb_dim])+_emb_decoder

                    # the attention weights
                    _attn_weight=tf.matmul(attn_d,tf.transpose(self.encoder_outputs,[0,2,1])) #batch,dec_timesteps,enc_timesteps
                    attn_weight=tf.nn.softmax(_attn_weight,axis=-1)#batch,dec_timesteps,enc_timesteps
                    # the conditional input
                    attns=tf.matmul(attn_weight,self.encoder_out)# batch,seq_len_target,dim

                    attn_c.append(attns)
                    emb_decoder=attn_h+attns
                    last_decoder_outputs=emb_decoder

            self.MAttenOut=emb_decoder


        """对文本摘要的topic的decoder端计算"""
        with tf.variable_scope("topic_decoder",reuse=reuse):
            # decoder端文本摘要的topic的embedding
            topic_decoder_outputs=self.topic_embbeding(abstract)
            topic_decoder_position=tf.nn.embedding_lookup(self.pos_emb,abstract_position)

            _topic_emb=tf.reduce_sum([topic_decoder_outputs,topic_decoder_position],axis=0)

            last_topic_outputs=topic_emb=_topic_emb

            for topic_layer in range(hps.con_layers):
                with tf.variable_scope("decoder_%d"%topic_layer),tf.device(self._next_device()):
                    topic_emb = tf.pad(topic_emb, paddings=[[0, 0], [padsize, padsize-1], [0, 0]], constant_values=self.PAD_emb)
                    topic_emb=tf.expand_dims(topic_emb,-1)
                    # the convolution
                    filter = tf.Variable(initial_value=tf.truncated_normal(self.filters, stddev=0.01), name="filter")
                    conv_h=tf.nn.conv2d(topic_emb,filter,strides=[1,1,1,1],padding="VALID")
                    conv_h=tf.reshape(conv_h,shape=[hps.batch_size,-1,hps.emb_dim*2])
                    # the gate linear unit
                    A,B=tf.split(conv_h,2,axis=2)
                    attn_h=tf.multiply(A,tf.nn.softmax(B))

                    # residual connections
                    attn_h=attn_h+last_topic_outputs

                    # attention mechanism
                    topic_emb=tf.reshape(attn_h,shape=[-1,hps.emb_dim]) # batch*seq_len,dim
                    attn_d=self.xw_plus_b(topic_emb,shapes=[hps.emb_dim, hps.emb_dim])
                    attn_d = tf.reshape(attn_d, shape=[hps.batch_size, -1, hps.emb_dim]) + _topic_emb

                    # the attention weights
                    _attn_weight1 = tf.matmul(attn_d, tf.transpose(self.encoder_outputs,[0, 2, 1]))  #batch,dec_timesteps,enc_timesteps
                    _attn_weight2=tf.matmul(attn_d,tf.transpose(self.topic_outputs,[0,2,1]))
                    _attn_weight=_attn_weight1+_attn_weight2
                    attn_weight = tf.nn.softmax(_attn_weight, axis=-1) #batch,dec_timesteps,enc_timesteps

                    # the conditional input
                    attns=tf.matmul(attn_weight,self.topic_out)# batch,seq_len_target,dim

                    topic_emb=attn_h+attns+attn_c[topic_layer]
                    last_topic_outputs=topic_emb

            self.TAttenOut = topic_emb# batch,seq_len_target,dim

    def xw_plus_b(self,emb,shapes):
        weight=tf.get_variable(name="weight",shape=shapes,dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01))
        bias=tf.get_variable(name='bias',shape=[shapes[1]],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.001))
        attn = tf.nn.xw_plus_b(emb, weight, bias)
        return attn

    def _BiasedProGen(self,reuse=False):
        """Biased Probability Generation"""
        hps=self._hps
        vsize = self._vsize
        with tf.variable_scope("bias_pro_gen",reuse=reuse),tf.device(self._next_device()):
            with tf.variable_scope("wordAtten"):
                # 词汇的atten机制
                MAtten=tf.reshape(self.MAttenOut,shape=[-1,hps.emb_dim])
                MAtten=self.xw_plus_b(MAtten,shapes=[hps.emb_dim,vsize])
                MAtten=tf.reshape(MAtten,shape=[hps.batch_size,-1,vsize])

            with tf.variable_scope("topicAtten"):
                # 主题词汇表的atten机制
                TAtten=tf.reshape(self.TAttenOut,shape=[-1,hps.emb_dim])
                TAtten=self.xw_plus_b(TAtten,shapes=[hps.emb_dim,vsize])
                TAtten=tf.reshape(TAtten,shape=[hps.batch_size,-1,vsize])

            _target=tf.exp(MAtten)+tf.multiply(tf.exp(TAtten),self.indicator)# batch,seq_len_tartget,vsize
            return _target

    def _build_graph(self):
        """只是为了建立tensorflow流程图"""
        hps = self._hps
        loss = []
        pad_id = self._vocab.WordToId(PAD_TOKEN)
        mask=tf.to_float(tf.not_equal(self.target,pad_id))
        self._Encoder()
        for t in range(hps.dec_timesteps):
            abstract = self.abstract[:,:t+1]
            abstract_position = self.abstract_position[:,:t+1]

            self._Decoder(abstract, abstract_position, reuse=(t != 0))
            logits = self._BiasedProGen(reuse=(t != 0))
            # softmax = tf.nn.softmax(logits, axis=-1, name=None)
            logits= tf.argmax(logits, axis=-1)
            logits=tf.expand_dims(logits[:,-1],1)
            logits=tf.cast(logits,tf.float32)
            target = self.target[:, t]
            loss1=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,logits=logits)
            loss += tf.reduce_sum(loss1 * mask[:, t])


    def _sample(self):
        """sampling from the distribution and compute loss"""
        hps=self._hps
        sample_words_list=[]
        loss=[]
        start_id=self._vocab.WordToId(PARAGRAPH_START)
        sample_words_list.append(tf.fill([hps.batch_size,1],start_id))
        end_id=self._vocab.WordToId(PARAGRAPH_END)
        self._Encoder()
        for t in range(hps.dec_timesteps):
            if t==0:
                abstract=tf.fill([hps.batch_size,1],start_id)
                abstract_position=tf.fill([hps.batch_size,1],0)
            else:
                abstract=sample_words
                abstract_position=sample_word_position
            self._Decoder(abstract,abstract_position,reuse=(t!=0))
            logits=self._BiasedProGen(reuse=(t!=0))
            softmax = tf.nn.softmax(logits, axis=-1, name=None)
            softmax=tf.expand_dims(softmax[:,-1,:],axis=1)
            softmax=tf.reshape(softmax,[hps.batch_size,-1])
            sample_word = tf.multinomial(tf.log(tf.clip_by_value(softmax, 1e-20, 1.0)), 1)
            sample_word=tf.cast(sample_word,dtype=tf.int32)
            sample_words_list.append(sample_word)
            sample_words=tf.concat(sample_words_list,1)

            sample_word_position=tf.concat([abstract_position,tf.fill([hps.batch_size,1],t)],1)
            loss.append(tf.log(tf.clip_by_value(softmax, 1e-20, 1.0)) * tf.one_hot(tf.identity(sample_word), self._vsize))

        sampled_captions=sample_words
        loss_out = tf.concat(loss,-1)
        return sampled_captions,loss_out

    def _greed_sample(self):
        """greedily selecting words"""
        hps=self._hps
        sample_words_list=[]
        start_id = self._vocab.WordToId(PARAGRAPH_START)
        sample_words_list.append(tf.fill([hps.batch_size,1],start_id))
        end_id = self._vocab.WordToId(PARAGRAPH_END)
        self._Encoder()
        for t in range(hps.dec_timesteps):
            if t == 0:
                abstract = tf.fill([hps.batch_size, 1], start_id)
                abstract_position = tf.fill([hps.batch_size, 1], 0)
            else:
                abstract = sample_words
                abstract_position = sample_word_position

            self._Decoder(abstract, abstract_position, reuse=(t != 0))
            logits = self._BiasedProGen(reuse=(t != 0))
            sample_word = tf.argmax(logits, axis=-1)
            sample_word=tf.expand_dims(sample_word[:,-1],1)
            sample_word=tf.cast(sample_word,dtype=tf.int32)
            sample_words_list.append(sample_word)
            sample_words = tf.concat(sample_words_list, 1)
            sample_word_position = tf.concat([abstract_position, tf.fill([hps.batch_size, 1], t)], 1)

        sampled_captions=sample_words
        return sampled_captions


    def _build_loss(self):
        """建立损失函数"""
        hps=self._hps
        loss=[]
        start_id = self._vocab.WordToId(PARAGRAPH_START)
        pad_id = self._vocab.WordToId(PAD_TOKEN)
        sample_caption=self.sample_caption
        mask = tf.to_float(tf.not_equal(sample_caption, pad_id))

        self._Encoder()
        for t in range(hps.dec_timesteps):
            abstract = sample_caption[:, :t + 1]
            if t == 0:
                abstract_position = tf.fill([hps.batch_size, 1], 0)
            else:
                abstract_position = tf.concat([abstract_position, tf.fill([hps.batch_size, 1], t)], 1)

            self._Decoder(abstract, abstract_position, reuse=(t != 0))
            logits = self._BiasedProGen(reuse=(t != 0))
            softmax = tf.nn.softmax(logits, axis=-1, name=None)
            softmax=tf.reduce_max(softmax,-1)
            softmax=tf.expand_dims(softmax[:,-1],1)
            loss1=tf.transpose(tf.log(tf.clip_by_value(softmax, 1e-20, 1.0)) * tf.one_hot(abstract[:, t], self._vsize),[1, 0])
            loss.append(tf.transpose(tf.multiply(loss1,  mask[:, t]), [1, 0]))

        loss_out = tf.concat(loss,-1)

        return loss_out

    def look_step(self,g,index,single, k1):
        t = tf.cond(tf.equal(single, self.topic_to_vocab[k1, 1]),
                    lambda: (True,k1),
                    lambda: (False,-1))
        return tf.logical_or(g, t[0]),tf.maximum(t[1],index), single, k1 + 1

    def _look(self,single):
        """返回single是否在主题词汇表中，并返回在主题词汇表中的index"""
        g = False
        k1 = 0
        index=-1
        g,index, *_ = tf.while_loop(
            cond=lambda g,index, single, k1: k1 < self._tsize,
            body=self.look_step,
            loop_vars=[g,index, single, k1]
        )
        return g,index

    def loop_step(self,k,topic, matrix):
        g,index=self._look(topic[k])
        matr = tf.cond(g,
                       lambda: tf.gather(self.topic_emb, [self.topic_to_vocab[index, 1]]),
                       lambda: tf.gather(self.vocab_emb, [topic[k]]))

        matr = tf.reshape(matr, [1, 1, -1])
        # matrix = tf.concat([matrix, matr], 0)
        matrix=matrix.write(k,matr)
        return k + 1,topic, matrix

    def topic_embbeding(self,topic):
        """条件embedding"""
        topic_shape=topic.get_shape()
        topics=tf.reshape(topic,[-1])
        k = tf.constant(1)
        size=topic_shape[0]*topic_shape[1]
        # start_id = self._vocab.WordToId(PARAGRAPH_START)
        # matrix = tf.nn.embedding_lookup(self.vocab_emb, [start_id])
        # matrix = tf.reshape(matrix, [1, 1, -1])
        loop_vars=[tf.constant(0),tf.reshape(topic,[-1]),tf.TensorArray(tf.float32,size=size)]
        _,_, matrix1 = tf.while_loop(
            cond=lambda k, *_: k < size,
            body=self.loop_step,
            loop_vars=loop_vars,
            # shape_invariants=[k.get_shape(),tf.TensorShape([None]), tf.TensorShape([None, 1, self._hps.emb_dim])]#对于shape可变的variable需要定义的
        )
        matrix1 = tf.reshape(matrix1.stack(), shape=[self._hps.batch_size, topic_shape[1], self._hps.emb_dim])
        return matrix1




