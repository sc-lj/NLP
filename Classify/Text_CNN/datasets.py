# coding:utf-8

import zipfile,threading
import string,re,time,json
import numpy as np
from collections import Counter
from multiprocessing import cpu_count,Pool,Process
from queue import Queue


class DealData(object):
    def __init__(self,arg,logger,max_sequence_length=None):
        self.arg=arg
        self.logger=logger
        if max_sequence_length:
            self.max_sequence_length = max_sequence_length
        else:self.max_sequence_length=arg.max_sequence_length

    def read_corpus(self,filename,batch_size=1):
        with open(filename,'r',encoding='utf-8') as f:
            data=f.readlines(batch_size)
            while data:
                labels=[]
                content=[]
                for da in data:
                    jsdata=json.loads(da)
                    label,one_line=jsdata['label'],jsdata['content']
                    labels.append(label)
                    content.append(one_line)
                yield labels,content
                data = f.readlines(batch_size)

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
        text_vector=np.zeros([self.max_sequence_length,self.vocab_length],dtype=np.float32)
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
        读取测试数据集
        :param bow_seq:
        :return:
        """
        self.vocab = []
        with open(arg.vocab_file, 'r') as f:
            da = f.readline()
            while da:
                self.vocab.append(da.split(' ')[0])
                da = f.readline()
        self.vocab_length=len(self.vocab)

        self.labels=set() # 标签集
        valid_cont_label = []  # 文本内容和标签
        for label,one_line in self.read_corpus(self.arg.valid_file):
            self.labels.add(label[0])
            valid_cont_label.append([one_line[0], label[0]])
        data=np.array(valid_cont_label)
        X,Y =np.transpose(data)

        labellist=list(self.labels)

        # 验证集的大小
        if bow_seq=='seq':
            x_dev_vector=self.seq_vector(X)
        else:
            x_dev_vector=self.bow_vector(X)
        label_index = list(map(labellist.index, Y))
        dev_y_array = np.zeros([len(Y), len(labellist)], dtype=np.float32)
        dev_y_array[list(range(len(Y))), label_index] = 1
        return x_dev_vector,dev_y_array

    def batch_iter(self,bow_seq):
        labellist = list(self.labels)

        for epoch in range(self.arg.num_epochs):
            print('epoch',epoch)
            for label,content in self.read_corpus(self.arg.train_file,self.arg.batch_size):
                if bow_seq=='seq':
                    x_train_vector=self.seq_vector(content)
                else:
                    x_train_vector=self.bow_vector(content)
                label_index = list(map(labellist.index,label))
                y_train_array = np.zeros([len(label),len(labellist)], dtype=np.float32)
                y_train_array[list(range(len(label))),label_index]=1
                yield x_train_vector,y_train_array


if __name__ == '__main__':
    from config import *
    arg = Argparse()
    logger=log_config(__file__)
    dealdata = DealData(arg,logger)
    x_dev_vector, y_dev_array = dealdata.slice_batch(bow_seq='seq')
    # for x_batch, y_batch in dealdata.batch_iter(x_train, y_train, bow_seq='seq'):
    #     print(y_batch)
