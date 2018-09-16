# coding:utf-8

import tensorflow as tf

class Abs():
    def __init__(self,emb_dim, hid_dim, vocab_size, q, c,batch_size,enc_timesteps,dec_timesteps, encoder_type='attention'):
        self.emb_dim=emb_dim
        self.hid_dim=hid_dim
        self.vocab_size=vocab_size
        self._q=q
        self._c=c
        self.encoder_type=encoder_type
        self.batch_size=batch_size
        self.dec_timesteps=dec_timesteps
        self.enc_timesteps=enc_timesteps


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

            decoder_embs=[tf.nn.embedding_lookup(self._E,t_t) for t_t in decoder_inputs]
            encoder_embs = [tf.nn.embedding_lookup(self._F, encoder_input) for encoder_input in encoder_inputs]

            if self.encoder_type == 'bow':
                xt_embs = tf.concat([tf.reshape(embs, shape=[-1, self.emb_dim, 1]) for embs in encoder_embs], 2)
                W_enc = tf.nn.xw_plus_b(tf.reduce_mean(xt_embs, 2), self._W,self.W_biase)

            elif self.encoder_type == 'attention':
                x_bar=[tf.reduce_mean(encoder_embs[max(0,i-self._q):min(self.enc_timesteps,i+self._q)],0) for i in range(self.enc_timesteps)]
                encoder_y=[]

            decoder_embs = tf.concat([tf.reshape(embs, shape=[-1, self.emb_dim, 1]) for embs in decoder_embs], 2)
            h = tf.nn.tanh(self._U * decoder_embs)
            y_t = tf.nn.softmax(self._V * h + W_enc)

