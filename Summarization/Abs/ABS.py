# coding:utf-8

import tensorflow as tf
import numpy as np

def sequence_loss_by_example(inputs,targets,loss_function,name=None):
    with tf.name_scope(values=inputs + targets , name=name,
                       default_name='sequence_loss_by_example'):
        log_perp_list = []
        for inp, target in zip(inputs, targets):
            crossent = loss_function(inp, target)
            log_perp_list.append(crossent)
        logperp=tf.add_n(log_perp_list)
    return logperp

class ABS():
    def __init__(self, vocab_size, batch_size,enc_timesteps,dec_timesteps,emb_dim=200, hid_dim=400,L=3,Q=2,C=5, encoder_type='attention'):
        self.emb_dim=emb_dim
        self.hid_dim=hid_dim
        self.vocab_size=vocab_size
        self._q=Q
        self._c=C
        self._l=L
        self.encoder_type=encoder_type
        self.batch_size=batch_size
        self.dec_timesteps=dec_timesteps
        self.enc_timesteps=enc_timesteps
        self.num_softmax_samples=4056


    def _add_placeholder(self):
        self._articles = tf.placeholder(tf.int32,[self.batch_size, self.enc_timesteps],name='articles')
        self._abstracts = tf.placeholder(tf.int32,[self.batch_size, self.dec_timesteps],name='abstracts')
        self._targets = tf.placeholder(tf.int32,[self.batch_size, self.dec_timesteps],name='targets')


    def _add_weight(self):
        with tf.variable_scope("embedding"), tf.device("/cpu:0"):
            # embedding
            self._E = tf.get_variable("E", shape=[self.vocab_size, self.emb_dim], dtype=tf.float32)
            self._F = tf.get_variable("F", shape=[self.vocab_size, self.hid_dim], dtype=tf.float32)

            # weight
            self._U = tf.get_variable("U", shape=[self.hid_dim, self._c * self.emb_dim], dtype=tf.float32)
            self._V = tf.get_variable("V", shape=[self.vocab_size, self.hid_dim], dtype=tf.float32)
            self._W = tf.get_variable("W", shape=[self.vocab_size, self.hid_dim], dtype=tf.float32)
            self.W_biase=tf.get_variable("w_biase",shape=[self.vocab_size],dtype=tf.float32)

            # Attention-based encoder
            if self.encoder_type == 'attention':
                self._P = tf.get_variable("P", shape=[self.hid_dim, self._c * self.emb_dim], dtype=tf.float32)



    def _add_abs(self):
        with tf.variable_scope("ABS"):
            encoder_inputs=tf.unstack(tf.transpose(self._articles))
            decoder_inputs = tf.unstack(tf.transpose(self._abstracts))
            targets = tf.unstack(tf.transpose(self._targets))

            self._add_weight()

            self.decoder_embs=[tf.nn.embedding_lookup(self._E,t_t) for t_t in decoder_inputs]
            self.encoder_embs = [tf.nn.embedding_lookup(self._F, encoder_input) for encoder_input in encoder_inputs]

            if self.encoder_type == 'bow':
                with tf.variable_scope("Bow"):
                    self.BowEncoder()
            elif self.encoder_type == 'attention':
                with tf.variable_scope("attention"):
                    self.AttentionEncoder()
            elif self.encoder_type=='conv':
                with tf.variable_scope("conv"):
                    self.ConvEncoder()
            else:
                raise("没有提供该%s方法,请输入bow，attention，conv中的一种"%(self.encoder_type))

            with tf.variable_scope("output_project"):
                w_t=tf.get_variable("w",shape=[self.vocab_size,self.vocab_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1e-4))
                v=tf.get_variable("v",shape=[self.vocab_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1e-4))

            with tf.variable_scope('loss'):
                def sampled_loss_func(inputs, labels):
                    with tf.device('/cpu:0'):  # Try gpu.
                        labels = tf.reshape(labels, [-1, 1])
                        return tf.nn.sampled_softmax_loss(
                            weights=w_t, biases=v, labels=labels, inputs=inputs,
                            num_sampled=self.num_softmax_samples, num_classes=self.vocab_size)

            inputs=self._label
            self.cost = tf.reduce_sum(sequence_loss_by_example(inputs, targets, sampled_loss_func))

    def _train(self):
        self.optimizer=tf.train.AdagradOptimizer(0.01)


    def ConvEncoder(self):
        """卷积encoder"""
        pass


    def BowEncoder(self):
        """BOW encoder"""
        xt_embs = tf.concat([tf.reshape(embs, shape=[-1, self.emb_dim, 1]) for embs in self.encoder_embs], 2)
        W_enc = tf.nn.xw_plus_b(tf.reduce_mean(xt_embs, 2), self._W, self.W_biase)
        outputs=[]
        for i in range(len(self.decoder_embs)):
            NLM_y = tf.concat(self.decoder_embs[max(0, i - self._c + 1):i], 1)
            h = tf.nn.tanh(tf.matmul(self._U, NLM_y, transpose_b=True))
            output = tf.nn.softmax(self._V * h + W_enc)
            outputs.append(output)
        self._label=outputs


    def AttentionEncoder(self):
        """注意力Encoder"""
        x_bar = [tf.reduce_sum(self.encoder_embs[max(0, i - self._q):min(self.enc_timesteps, i + self._q)], 0) for i in
                 range(self.enc_timesteps)]
        x_bar = [tf.divide(bar, self._q) for bar in x_bar]
        x_bar = tf.concat([tf.reshape(xbar, shape=[self.batch_size, self.hid_dim, 1]) for xbar in x_bar], 2)

        outputs=[]
        for i in range(len(self.decoder_embs)):
            encoder_y = self.decoder_embs[max(0, i - self._c + 1):i]
            encoder_y = tf.concat(encoder_y, 1)
            enc_exp = tf.matmul(self._P, encoder_y, transpose_b=True)
            lowerp = tf.concat([tf.reduce_sum(tf.multiply(encoder_emb, enc_exp)) for encoder_emb in self.encoder_embs], 1)
            lowerp = tf.nn.softmax(lowerp)
            W_encs = tf.reduce_sum(tf.multiply(x_bar, tf.reshape(lowerp, shape=[self.batch_size, 1, -1])), 2)
            W_enc = tf.nn.xw_plus_b(W_encs, self._W, self.W_biase)

            NLM_y = tf.concat(self.decoder_embs[max(0, i - self._c + 1):i], 1)
            h = tf.nn.tanh(tf.matmul(self._U , NLM_y,transpose_b=True))
            output = tf.nn.softmax(self._V * h + W_enc)
            outputs.append(output)

        self._label=outputs




