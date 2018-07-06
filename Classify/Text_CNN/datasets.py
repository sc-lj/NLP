# coding:utf-8

import json,time,random
import numpy as np
import datetime


class DealData(object):
    def __init__(self,arg,logger,max_sequence_length=None):
        self.arg=arg
        self.logger=logger
        if max_sequence_length:
            self.max_sequence_length = max_sequence_length
        else:self.max_sequence_length=arg.max_sequence_length
        self.labels=['汽车', '商业', '健康', '体育', 'IT', '旅游', '娱乐文化', '教育']
        self.gene_dict()

    def read_corpus(self,filename,batch_size=1):
        with open(filename,'r',encoding='utf-8') as f:
            data=f.readline()
            j=0
            labels = []
            contents = []
            while data:
                j+=1
                jsdata=json.loads(data)
                label,one_line=jsdata['label'],jsdata['content']
                labels.append(label)
                contents.append(one_line)
                if j==int(batch_size):
                    yield labels,contents
                    j = 0
                    labels = []
                    contents = []
                data = f.readline()

    def gene_dict(self):
        """
        生成数值和字的映射
        :return:
        """
        self.vocab = []
        with open(self.arg.vocab_file, 'r',encoding='utf-8') as f:
            da = f.readline()
            while da:
                self.vocab.append(da.split(' ')[0])
                da = f.readline()
        self.vocab_length=len(self.vocab)

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
                if j>=self.max_sequence_length:
                    break
        return text_vector

    def seq_vector(self,text_list):
        """
        生成seq-CNN模型需要的向量
        :param text_list: 单个文本或者多个文本组成的list
        :return:
        """
        lines=np.array(text_list)
        if len(lines)==1:
            text_vectors=self.single_seq_vector(lines[0])
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
        text_vector = np.zeros([self.max_sequence_length, len(self.vocab)], dtype=np.float32)
        # 将没有在词汇表中的字用零代替
        for i in range(0, line_len-num):
            for word in line[i:i + num]:
                if word in self.vocab:
                    index = self.word_id[word]
                    text_vector[i][index] = 1
            if i+1>= self.max_sequence_length:
                break
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
            text_vectors=self.single_bow_line(lines[0],num)
        else:
            text_vectors=[]
            for line in lines:
                text_vector=self.single_bow_line(line,num)
                text_vectors.append(text_vector)
        return text_vectors

    def read_batch(self,filename=None,bow_seq='seq',batch_size=100):
        """
        默认读取测试数据集
        :param bow_seq:'seq'or 'bow'
        :return:
        """
        if not filename:
            filename=self.arg.valid_file
            if random.randint(0,6)==0:
                self.shuffle(filename)
        for label,one_line in self.read_corpus(filename=filename,batch_size=batch_size):
            # # 验证集的大小
            if bow_seq=='seq':
                x_vector=self.seq_vector(one_line)
            else:
                x_vector=self.bow_vector(one_line)
            label_index = list(map(self.labels.index, label))
            y_array = np.zeros([len(label), len(self.labels)], dtype=np.float32)
            y_array[list(range(len(label))), label_index] = 1
            yield x_vector,y_array

    def shuffle(self,filename):
        f1 = open(filename, 'r',encoding='utf-8')
        data = f1.readlines()
        f1.close()
        length = len(data)
        index = list(range(length))
        random.shuffle(index)
        f = open(filename, 'w',encoding='utf-8')
        for i in index:
            f.write(data[i])
        f.close()

    def batch_iter(self,bow_seq):
        for epoch in range(self.arg.num_epochs):
            self.shuffle(self.arg.train_file)
            time_str = datetime.datetime.now().isoformat()
            print("{}: epoch {}".format(time_str,epoch ))
            for x_vector, y_array in self.read_batch(self.arg.train_file,batch_size=self.arg.batch_size,bow_seq=bow_seq):
                yield x_vector, y_array


if __name__ == '__main__':
    from config import *
    arg = Argparse()
    logger=log_config(__file__)
    dealdata = DealData(arg,logger)
    # for x_dev_vector, y_dev_array in dealdata.read_batch(filename=arg.valid_file,bow_seq='seq',batch_size=1000):
    #     print(y_dev_array)
    #     time.sleep(5)
    # for x_batch, y_batch in dealdata.batch_iter(bow_seq='bow'):
    #     print(y_batch)
    valid_data = dealdata.read_batch(bow_seq='seq', batch_size=1000)
    x_dev_vector, y_dev_array = valid_data.__next__()
    print(y_dev_array)



