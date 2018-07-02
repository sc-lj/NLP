# coding:utf-8

import zipfile,threading
import string,re,time,json
import numpy as np
from collections import Counter
from multiprocessing import cpu_count,Pool,Process
from queue import Queue


class DealData(object):
    def __init__(self,arg,logger):
        self.arg=arg
        self.logger=logger
        self.max_sequence_length = 0

        self.labels = set()  # 标签集
        self.vocab = set()  # 词汇表
        self.cont_label = []  # 文本内容和标签
        self.word_freq = Counter()  # 词频表,Counter能对key进行累加

        # 词向量的长度
        self.vector_length=len(self.vocab)
        self.lable_length=len(self.labels)

    def read_corpus(self,filename):
        with open(filename,'r',encoding='utf-8') as f:
            data=f.readline()
            while data:
                jsdata=json.loads(data)
                label,one_line=jsdata['label'],jsdata['content']
                self.labels.add(label)
                self.vocab.update(set(one_line))
                self.cont_label.append([one_line,label])
                yield label,one_line
                data = f.readline()

    def gene_dict(self):
        """
        生成数值和字的映射
        :return:
        """
        self.word_id=dict(zip(self.vocab,range(len(self.vocab))))
        self.id_word=dict(zip(range(len(self.vocab)),self.vocab))

    def single_seq_vector(self,line):
        """
        针对单个文本生成向量
        :param line: 单个文本
        :return:
        """
        text_vector=np.zeros([self.max_sequence_length,self.vector_length],dtype=np.float32)
        j=0
        # 将没有在词汇表中的字用零代替
        for word in line:
            if word in list(self.vocab):
                index=self.word_id[word]
                text_vector[j][index] = 1
            j+=1
        return text_vector

    def seq_vector(self,text_list):
        """
        生成seq-CNN模型需要的向量
        :param text_list: 单个文本或者多个文本组成的list
        :return:
        """
        lines=np.array(text_list)
        if len(lines)==1:
            text_vectors=self.single_seq_vector(lines)
        else:
            text_vectors=[]
            for line in lines:
                text_vectors.append(self.single_seq_vector(line))

        return text_vectors

    def single_bow_line(self,line,num):
        """
        这是针对单个文本处理
        :param line: 单个文本
        :param num:
        :return:
        """
        line_len = len(line)
        text_vector = np.zeros([self.max_sequence_length- num + 1, len(self.vocab)], dtype=np.float32)
        # 将没有在词汇表中的字用零代替
        for i in range(0, line_len - num):
            for word in line[i:i + num]:
                if word in self.vocab:
                    index = self.word_id[word]
                    text_vector[i][index] = 1
        return text_vector

    def bow_vector(self,text_list,num=3):
        """
        生成bow-CNN模型需要的向量，这是针对多个文本集
        :param text_list: 单个文本的字串,是list形式
        :param num: 每个向量包含相邻的字，默认为3
        :return:
        """
        lines=np.array(text_list)

        if len(lines)==1:
            text_vectors=self.single_bow_line(lines,num)
        else:
            text_vectors=[]
            for line in lines:
                text_vector=self.single_bow_line(line,num)
                text_vectors.append(text_vector)
        return text_vectors

    def slice_batch(self,bow_seq='seq'):
        """
        切分数据集
        :param bow_seq:
        :return:
        """

        for label,one_line in self.read_corpus():
            if len(one_line)<=20:
                continue

        labellist=list(self.labels)
        data_size=len(self.cont_label)
        data=np.array(self.cont_label)
        X,Y =np.transpose(data)

        # 验证集的大小
        dev_data_size = -1 * int(self.arg.dev_sample_percent * data_size)
        x_dev, x_train = X[dev_data_size:],X[:dev_data_size]#验证集和训练集
        y_dev,y_train=Y[dev_data_size:],Y[:dev_data_size]
        if bow_seq=='seq':
            x_dev_vector=self.seq_vector(x_dev)
        else:
            x_dev_vector=self.bow_vector(x_dev)
        label_index = list(map(labellist.index, y_dev))
        dev_y_array = np.zeros([len(y_dev), len(labellist)], dtype=np.float32)
        dev_y_array[list(range(len(y_dev))), label_index] = 1
        return x_train,y_train,x_dev_vector,dev_y_array

    def batch_iter(self,x_train,y_train,bow_seq,shuffle=True):
        labellist = list(self.labels)
        # 重新计算数据大小
        data_size=len(x_train)

        for epoch in range(self.arg.num_epochs):
            print('epoch',epoch)
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                x_train = x_train[list(shuffle_indices)]
                y_train = y_train[list(shuffle_indices)]

            for num in range(0,data_size,self.arg.batch_size):
                end_index=num+self.arg.batch_size
                if data_size< end_index:
                    end_index = data_size
                conts=x_train[num:end_index]
                labels=y_train[num:end_index]
                if bow_seq=='seq':
                    x_train_vector=self.seq_vector(conts)
                else:
                    x_train_vector=self.bow_vector(conts)
                label_index = list(map(labellist.index,labels))
                y_train_array = np.zeros([len(labels),len(labellist)], dtype=np.float32)
                y_train_array[list(range(len(labels))),label_index]=1
                yield x_train_vector,y_train_array


if __name__ == '__main__':
    from config import *
    arg = Argparse()
    logger=log_config(__file__)
    dealdata = DealData(arg,logger)
    # x_train, y_train, x_dev_vector, y_dev_array = dealdata.slice_batch(bow_seq='seq')
    # for x_batch, y_batch in dealdata.batch_iter(x_train, y_train, bow_seq='seq'):
    #     print(y_batch)

