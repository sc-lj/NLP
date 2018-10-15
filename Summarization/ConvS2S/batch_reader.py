# coding:utf-8

import tensorflow as tf
from random import shuffle
from queue import Queue
from collections import namedtuple
from Data import *
from threading import Thread
import time
import numpy as np

ModelInput=namedtuple("ModelInput",'enc_input enc_position dec_input target enc_len dec_len')

BUCKET_CACHE_BATCH = 100
QUEUE_NUM_BATCH = 100

class Batcher():
    def __init__(self,vocab,tvocab,hps,max_article_length,max_abstract_length,bucketing=True,truncate_input=False):
        self._vocab=vocab # 词汇表
        self._tvocab=tvocab # 主题词汇表
        self._hps=hps
        self.max_article_length=max_article_length
        self.max_abstract_lenght=max_abstract_length
        self.bucket=bucketing
        self.truncate_input=truncate_input
        self.input_queue=Queue(QUEUE_NUM_BATCH*self._hps.batch_size)
        self.bucket_input_queue=Queue(QUEUE_NUM_BATCH)
        self._input_threads=[]
        for _ in range(16):
            self._input_threads.append(Thread(target=self._InputQueue))
            self._input_threads[-1].daemon=True
            self._input_threads[-1].start()

        self._bucket_threads=[]
        for _ in range(8):
            self._bucket_threads.append(Thread(target=self._BucketInputQueue))
            self._bucket_threads[-1].daemon=True
            self._bucket_threads[-1].start()

        self.watch_thread=Thread(target=self._WatchThreads)
        self.watch_thread.daemon=True
        self.watch_thread.start()


    def NextBatch(self):
        hps=self._hps
        enc_input_batch=np.zeros([hps.batch_size,hps.enc_timesteps],dtype=np.int32)
        enc_position_batch=np.zeros([hps.batch_size,hps.enc_timesteps],dtype=np.int32)
        enc_lens=np.zeros([hps.batch_size],dtype=np.int32)

        dec_input_batch=np.zeros([hps.batch_size,hps.dec_timesteps],dtype=np.int32)
        dec_lens=np.zeros([hps.batch_size],dtype=np.int32)
        target_batch=np.zeros([hps.batch_size,])

        buckets=self.bucket_input_queue.get()
        for i in range(hps.batch_size):
            enc_input,enc_position,enc_topic,dec_input,target,enc_len,dec_len=buckets[i]
            enc_input_batch[i,:]=enc_input[:]
            enc_position_batch[i,:]=enc_position
            enc_lens[i]=enc_len

            dec_input_batch[i,:]=dec_input
            dec_lens[i]=dec_len
            target_batch[i]=target

        return (enc_input_batch,enc_position_batch,enc_lens,dec_input_batch,dec_lens,target_batch)


    def _WatchThreads(self):
        while True:
            time.sleep(66)
            input_threads=[]
            for t in self._input_threads:
                if t.is_alive():
                    input_threads.append(t)
                else:
                    tf.logging.warning("InputQueue线程宕掉了，需要重启")
                    new_t=Thread(target=self._InputQueue)
                    input_threads.append(new_t)
                    input_threads[-1].daemon=True
                    input_threads[-1].start()
            self._input_threads=input_threads

            bucket_threads=[]
            for t in self._bucket_threads:
                if t.is_alive():
                    bucket_threads.append(t)
                else:
                    tf.logging.warning("BucketInputQueue线程宕掉了，需要重启")
                    new_t=Thread(target=self._BucketInputQueue)
                    bucket_threads.append(new_t)
                    bucket_threads[-1].daemon=True
                    bucket_threads[-1].start()
            self._bucket_threads=bucket_threads


    def _BucketInputQueue(self):
        hps=self._hps
        while True:
            inputs=[]
            for _ in range(hps.batch_size*QUEUE_NUM_BATCH):
                inputs.append(self.input_queue.get())
            if self.bucket:
                inputs=sorted(inputs,key=lambda x:x.enc_len)

            batches=[]
            for i in range(0,len(inputs),hps.batch_size):
                batches.append(inputs[i:i+hps.batch_size])

            shuffle(batches)
            for b in batches:
                self.bucket_input_queue.put(b)


    def _InputQueue(self):
        hps=self._hps
        start_id=self._vocab.WordToId(PARAGRAPH_START)
        end_id=self._vocab.WordToId(PARAGRAPH_END)
        pad_id=self._vocab.WordToId(PAD_TOKEN)
        input_gen=ExampleGen()
        while True:
            article,abstract=next(input_gen)
            article_sentence=[sent.strip() for sent in ToSentences(article,include_token=False)]
            abstract_sentence=[sent.strip() for sent in ToSentences(abstract,include_token=False)]

            enc_input=[start_id]
            enc_position=[0]

            dec_input=[]

            for i in range(min(self.max_article_length,len(article_sentence))):
                enc_input+=GetWordIds(article_sentence[i],self._vocab)
                enc_position.append(i)

            for i in range(min(self.max_abstract_lenght,len(abstract_sentence))):
                dec_input+=GetWordIds(abstract_sentence[i],self._vocab)

            if (len(enc_input)<hps.min_input_len or len(dec_input)<hps.min_input_len):
                tf.logging.warning("丢掉文本的长度或者摘要的长度太短的样本,\nenc:%d\ndec:%d",len(enc_input),len(dec_input))
                continue

            if not self.truncate_input:
                if (len(enc_input)>hps.enc_timesteps or len(dec_input)>hps.dec_timesteps):
                    tf.logging.warning("丢掉文本长度或者摘要长度过长的样本,\nenc:%d\ndec:%d",len(enc_input),len(dec_input))
                    continue
            else:
                if len(enc_input)>hps.enc_timesteps:
                    enc_input=enc_input[:hps.enc_timesteps]
                    enc_position=enc_position[:hps.enc_timesteps]
                if len(dec_input)>hps.dec_timesteps:
                    dec_input=dec_input[:hps.dec_timesteps]

            target=dec_input[1:]
            target.append(end_id)

            enc_input_len=len(enc_input)
            dec_output_len=len(target)

            while len(enc_input)<hps.enc_timesteps:
                enc_input.append(pad_id)
                enc_position.append(len(enc_position))

            while len(dec_input)<hps.dec_timesteps:
                dec_input.append(pad_id)

            while len(target)<hps.dec_timesteps:
                target.append(pad_id)

            element=ModelInput(enc_input,enc_position,dec_input,target,enc_input_len,dec_output_len)
            self.input_queue.put(element)












