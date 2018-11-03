# coding:utf-8

import tensorflow as tf
from threading import Thread
from queue import Queue
from genVocab import *
from random import shuffle
from collections import namedtuple
import time
import numpy as np

tf.flags.DEFINE_string("source_file","../data/xinhua_source.txt","file of source")
tf.flags.DEFINE_string("target_file","../data/xinhua_target.txt",'file of target')
tf.flags.DEFINE_string("corpus_file","","the file of corpus")
tf.flags.DEFINE_integer("epoch",5,"the number of the epoch")
FLAGS=tf.flags.FLAGS

ModelInput=namedtuple("ModelInput","source target source_len")
QUEUE_NUM_BATCH=5

class Dataset():
    def __init__(self,hps,vocab):
        self.Glob_var=False
        self._hps=hps
        self.vocab=vocab
        self.inputQueue=Queue(QUEUE_NUM_BATCH*hps.batch_size)
        self.bucket_input_queue=Queue(QUEUE_NUM_BATCH)

        self._input_threads = []
        for _ in range(16):
            self._input_threads.append(Thread(target=self._InputQueue))
            self._input_threads[-1].daemon = True
            self._input_threads[-1].start()

        self._bucket_threads = []
        for _ in range(8):
            self._bucket_threads.append(Thread(target=self._BucketInputQueue))
            self._bucket_threads[-1].daemon = True
            self._bucket_threads[-1].start()

        self.watch_thread = Thread(target=self._WatchThreads)
        self.watch_thread.daemon = True
        self.watch_thread.start()

    def NextBatch(self,batch=None):
        hps = self._hps
        if batch is None:
            batch_size=hps.batch_size
        else:batch_size=batch

        source_input_batch=np.zeros([batch_size,hps.enc_timesteps],dtype=np.int32)
        target_input_batch=np.zeros([batch_size,hps.enc_timesteps],dtype=np.int32)
        source_input_len=np.zeros([batch_size],dtype=np.int32)

        buckets=self.bucket_input_queue.get()
        for i in range(batch_size):
            source,target,source_len=buckets[i]
            source_input_batch[i,:]=source[:]
            target_input_batch[i,:]=target[:]
            source_input_len[i]=source_len

        return (source_input_batch,target_input_batch,source_input_len)


    def _WatchThreads(self):
        while not self.Glob_var and self.bucket_input_queue.empty():
            time.sleep(66)
            input_threads = []
            for t in self._input_threads:
                if t.is_alive():
                    input_threads.append(t)
                else:
                    tf.logging.warning("InputQueue线程宕掉了，需要重启")
                    new_t = Thread(target=self._InputQueue)
                    input_threads.append(new_t)
                    input_threads[-1].daemon = True
                    input_threads[-1].start()
            self._input_threads = input_threads

            bucket_threads = []
            for t in self._bucket_threads:
                if t.is_alive():
                    bucket_threads.append(t)
                else:
                    tf.logging.warning("BucketInputQueue线程宕掉了，需要重启")
                    new_t = Thread(target=self._BucketInputQueue)
                    bucket_threads.append(new_t)
                    bucket_threads[-1].daemon = True
                    bucket_threads[-1].start()
            self._bucket_threads = bucket_threads


    def _InputQueue(self):
        hps=self._hps
        source = open(FLAGS.source_file, 'r')
        target = open(FLAGS.target_file, 'r')

        source_line = source.readlines()
        target_line = target.readlines()

        assert len(source_line) == len(target_line), ("源文件和标签文件的长度不相等！！")

        length = list(range(len(source_line)))
        j = 0
        while j < FLAGS.epoch:
            shuffle(length)
            for i in length:
                source_piece = source_line[i].split()
                source_piece = GetWordIds(self.vocab, source_piece)

                target_piece = target_line[i].split()
                length=len(source_piece)

                element=ModelInput(source_piece,target_piece,length)
                self.inputQueue.put(element)
            j += 1

        self.Glob_var=True

    def _BucketInputQueue(self):
        hps = self._hps
        while not self.Glob_var and self.inputQueue.empty():
            inputs = []
            for _ in range(hps.batch_size * QUEUE_NUM_BATCH):
                inputs.append(self.inputQueue.get())

            batches = []
            for i in range(0, len(inputs), hps.batch_size):
                batches.append(inputs[i:i + hps.batch_size])

            shuffle(batches)
            for b in batches:
                self.bucket_input_queue.put(b)



if __name__ == '__main__':
    vocab=Vocab()
    dt=Dataset('',vocab)
    dt._InputQueue()
