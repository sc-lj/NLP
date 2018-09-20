# coding:utf-8

import tensorflow as tf
import numpy as np
from collections import namedtuple

HParams=namedtuple("HParams","batch_size mode num_softmax_samples min_lr lr hid_dim emb_dim max_grad_norm Q C dec_timesteps enc_timesteps")


def sequence_loss_by_example(inputs,targets,weight,loss_function,name=None):
    with tf.name_scope(values=inputs + targets , name=name,
                       default_name='sequence_loss_by_example'):
        log_perp_list = []
        for inp, target in zip(inputs, targets):
            crossent = loss_function(inp, target)
            log_perp_list.append(crossent*weight)
        logperp=tf.add_n(log_perp_list)
    return logperp

class ABS():
    def __init__(self, hps,vocab_size,L=3, encoder_type='attention'):
        self._hps=hps
        self.vocab_size=vocab_size
        self._l=L
        self.encoder_type=encoder_type
        self.num_softmax_samples=4056


    def run_train_step(self,sess, article_batch, abstract_batch, targets, article_lens,
                abstract_lens, loss_weights):
        to_return = [self._train, self._summeries, self.cost, self.global_step]
        return sess.run(to_return,
                        feed_dict={self._articles: article_batch,
                                   self._abstracts: abstract_batch,
                                   self._targets: targets,
                                   self._articles_len: article_lens,
                                   self._abstracts_len: abstract_lens,
                                   self._loss_weight: loss_weights})


    def _add_placeholder(self):
        hps = self._hps
        self._articles = tf.placeholder(tf.int32,[hps.batch_size, hps.enc_timesteps],name='articles')
        self._articles_len=tf.placeholder(tf.int32,[hps.batch_size],name="article_len")
        self._abstracts = tf.placeholder(tf.int32,[hps.batch_size, hps.dec_timesteps],name='abstracts')
        self._abstracts_len=tf.placeholder(tf.int32,[hps.batch_size],name="abstract_len")
        self._targets = tf.placeholder(tf.int32,[hps.batch_size, hps.dec_timesteps],name='targets')
        self._loss_weight=tf.placeholder(tf.int32,[hps.batch_size,hps.dec_timesteps],name="loss_weight")

    def _add_weight(self):
        hps=self._hps
        with tf.variable_scope("embedding"), tf.device("/cpu:0"):
            # embedding
            self._E = tf.get_variable("E", shape=[self.vocab_size, hps.emb_dim], dtype=tf.float32)
            self._F = tf.get_variable("F", shape=[self.vocab_size, hps.hid_dim], dtype=tf.float32)

            # weight
            self._U = tf.get_variable("U", shape=[hps.hid_dim, hps.C * hps.emb_dim], dtype=tf.float32)
            self._V = tf.get_variable("V", shape=[self.vocab_size, hps.hid_dim], dtype=tf.float32)
            self._W = tf.get_variable("W", shape=[self.vocab_size, hps.hid_dim], dtype=tf.float32)
            self.W_biase=tf.get_variable("w_biase",shape=[self.vocab_size],dtype=tf.float32)

            # Attention-based encoder
            if self.encoder_type == 'attention':
                self._P = tf.get_variable("P", shape=[hps.hid_dim, hps.C * hps.emb_dim], dtype=tf.float32)



    def _add_abs(self):
        hps=self._hps
        with tf.variable_scope("ABS"):
            encoder_inputs=tf.unstack(tf.transpose(self._articles))
            decoder_inputs = tf.unstack(tf.transpose(self._abstracts))
            targets = tf.unstack(tf.transpose(self._targets))
            weight=tf.unstack(tf.transpose(self._loss_weight))

            self._add_weight()

            self.decoder_embs=[tf.nn.embedding_lookup(self._E,t_t) for t_t in decoder_inputs]
            self.encoder_embs = [tf.nn.embedding_lookup(self._F, encoder_input) for encoder_input in encoder_inputs]

            if self.encoder_type == 'bow':
                with tf.variable_scope("Bow"):
                    xt_embs = tf.concat([tf.reshape(embs, shape=[-1, self._hps.emb_dim, 1]) for embs in self.encoder_embs], 2)
                    self.W_enc = tf.nn.xw_plus_b(tf.reduce_mean(xt_embs, 2), self._W, self.W_biase)
                    if hps.mode=="train":
                        outputs=[]
                        for i in range(len(self.decoder_embs)):
                            encoder=self.decoder_embs[max(0, i - hps.C + 1):i]
                            output=self.BowEncoder(encoder)
                            outputs.append(output)
                    elif hps.mode=='decode':
                        outputs=[self.BowEncoder(self.decoder_embs)]
                    else:raise ValueError("没有该mode")
                    self._label=outputs

            elif self.encoder_type == 'attention':
                with tf.variable_scope("attention"):
                    x_bar = [tf.reduce_sum(self.encoder_embs[max(0, i - hps.Q):min(hps.enc_timesteps, i + hps.Q)], 0) for i in range(hps.enc_timesteps)]
                    x_bar = [tf.divide(bar, hps.Q) for bar in x_bar]
                    self.x_bar = tf.concat( [tf.reshape(xbar, shape=[self._hps.batch_size, self._hps.hid_dim, 1]) for xbar in x_bar], 2)
                    if hps.mode=='train':
                        outputs=[]
                        for i in range(len(self.decoder_embs)):
                            encoder_y=self.decoder_embs[max(0, i - hps.C + 1):i]
                            output=self.AttentionEncoder(encoder_y)
                            outputs.append(output)
                    elif hps.mode=='decode':
                        outputs =[self.AttentionEncoder(self.decoder_embs)]
                    else:raise ValueError("没有该mode")
                    self._label=outputs
            elif self.encoder_type=='conv':
                with tf.variable_scope("conv"):
                    self.ConvEncoder()
            else:
                raise("没有提供该%s方法,请输入bow，attention，conv中的一种"%(self.encoder_type))

            with tf.variable_scope("output_project"):
                w_t=tf.get_variable("w",shape=[hps.hid_dim,self.vocab_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1e-4))
                v=tf.get_variable("v",shape=[self.vocab_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1e-4))

            with tf.variable_scope('loss'):
                def sampled_loss_func(inputs, labels):
                    with tf.device('/cpu:0'):
                        labels = tf.reshape(labels, [-1, 1])
                        return tf.nn.sampled_softmax_loss(
                            weights=w_t, biases=v, labels=labels, inputs=inputs,
                            num_sampled=self.num_softmax_samples, num_classes=self.vocab_size)

            self.last_state=self._label[-1]
            if hps.mode=="decode":
                with tf.variable_scope("output"):
                    best_outputs=[tf.argmax(output,1) for output in self._label]
                    self.outputs=tf.concat([tf.reshape(output,shape=[hps.batch_size,1]) for output in best_outputs],axis=1)
                    self.topk_log_probs,self.topk_ids=tf.nn.top_k(tf.log(self._label[-1]),2*hps.batch_size)

            if hps.mode=="train":
                # 训练的时候，使用负采样
                self.cost = tf.reduce_sum(sequence_loss_by_example(self._label, targets,weight, sampled_loss_func))
            else:
                self.cost=tf.contrib.legacy_seq2seq.sequence_loss(self._label,targets,weight)

    def topk(self,sess,enc_inputs,enc_len,abstracts,abstracts_len):
        feed_dict={
            self._articles:enc_inputs,
            self._articles_len:enc_len,
            self._abstracts:np.array(abstracts),
            self._abstracts_len:abstracts_len
        }
        results=sess.run([self.topk_ids,self.topk_log_probs],feed_dict=feed_dict)
        ids, probs = results[0], results[1]
        return ids,probs

    def _train(self):
        hps=self._hps
        self._lr=tf.maximum(hps.min_lr,tf.train.exponential_decay(hps.lr,self.global_step,3000,0.98))
        vars=tf.trainable_variables()
        grads,global_norm=tf.clip_by_global_norm(tf.gradients(self.cost,vars),hps.max_grad_norm)
        optimizer=tf.train.AdagradOptimizer(self._lr)
        self.train_op=optimizer.apply_gradients(zip(grads,vars),self.global_step,name="train_op")


    def ConvEncoder(self):
        """卷积encoder"""
        pass


    def BowEncoder(self,encoder):
        """BOW encoder"""
        NLM_y = tf.concat(encoder, 1)
        h = tf.nn.tanh(tf.matmul(self._U, NLM_y, transpose_b=True))
        output = tf.nn.softmax(self._V * h + self.W_enc)
        return output

    def AttentionEncoder(self,encoder):
        """注意力Encoder"""
        encoder_y = tf.concat(encoder, 1)
        enc_exp = tf.matmul(self._P, encoder_y, transpose_b=True)
        lowerp = tf.concat([tf.reduce_sum(tf.multiply(encoder_emb, enc_exp)) for encoder_emb in self.encoder_embs], 1)
        lowerp = tf.nn.softmax(lowerp)
        W_encs = tf.reduce_sum(tf.multiply(self.x_bar, tf.reshape(lowerp, shape=[self._hps.batch_size, 1, -1])), 2)
        W_enc = tf.nn.xw_plus_b(W_encs, self._W, self.W_biase)

        NLM_y = tf.concat(encoder, 1)
        h = tf.nn.tanh(tf.matmul(self._U , NLM_y,transpose_b=True))
        output = tf.nn.softmax(self._V * h + W_enc)
        return output


    def build_graph(self):
        self._add_placeholder()
        self._add_abs()
        self.global_step=tf.Variable(0,trainable=False,name="global_step")
        self._summeries=tf.summary.merge_all()


