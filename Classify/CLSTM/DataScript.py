# coding:utf-8

import os,sys
syspath=os.path.abspath(__file__)
while True:
    sys.path.append(syspath)
    try:
        import GloConfi
        break
    except:
        syspath=os.path.dirname(syspath)

from Config import *
import json,random
import numpy as np
import datetime

class Data():
    def __init__(self, arg, logger):
        self.arg = arg
        self.logger = logger
        self.max_sequence_length = arg.max_sequence_length
        self.labels = ['汽车', '商业', '健康', '体育', 'IT', '旅游', '娱乐文化', '教育']
        self.gene_dict()

    def gene_dict(self):
        """
        生成数值和字的映射
        :return:
        """
        # 用<NONE>表示词汇中没有出现的词
        self.vocab = ['<NONE>']
        with open(self.arg.vocab_file, 'r',encoding='utf-8') as f:
            da = f.readline()
            while da:
                self.vocab.append(da.split(' ')[0])
                da = f.readline()
        self.vocab_length=len(self.vocab)

        self.word_id=dict(zip(self.vocab,range(len(self.vocab))))
        self.id_word=dict(zip(range(len(self.vocab)),self.vocab))


    def read_corpus(self, filename, batch_size):
        with open(filename, 'r', encoding='utf-8') as f:
            data = f.readline()
            j = 0
            labels = []
            contents = []
            while data:
                j += 1
                jsdata = json.loads(data)
                label, one_line = jsdata['label'], jsdata['content']
                labels.append(label)
                contents.append(one_line)
                if j == int(batch_size):
                    yield labels, contents
                    j = 0
                    labels = []
                    contents = []
                data = f.readline()

    def shuffle(self,filename):
        f1 = open(filename, 'r')
        data = f1.readlines()
        f1.close()
        length = len(data)
        index = list(range(length))
        random.shuffle(index)
        f = open(filename, 'w')
        for i in index:
            f.write(data[i])
        f.close()

    def single_line(self,line):
        """
        这是针对单个文本处理
        :param line: 单个文本
        :return:
        """
        text_vector = []
        # 将没有在词汇表中的字用零代替
        j=0
        for word in line:
            if word in self.vocab:
                index = self.word_id[word]
            else:
                index = self.word_id['<NONE>']
            text_vector.append(index)
            j += 1
            if j>= self.max_sequence_length:
                break

        while self.max_sequence_length-j:
            index = self.word_id['<NONE>']
            text_vector.append(index)
            j+=1
        return text_vector

    def word_index(self,lines):
        """
        生成多个文本的向量矩阵，这是针对多个文本集
        :param text_list: 单个文本的字串,是list形式
        :param num: 每个向量包含相邻的字，默认为3
        :return:
        """
        # lines=np.array(lines)
        if len(lines)==1:
            text_vectors=self.single_line(lines[0])
        else:
            text_vectors=[]
            for line in lines:
                text_vector=self.single_line(line)
                text_vectors.append(text_vector)
        return text_vectors

    def gen_batch(self,filename,batch_size=2):
        for labels, contents in self.read_corpus(filename,batch_size):
            label_index = list(map(self.labels.index, labels))
            y_array = np.zeros([len(labels), len(self.labels)], dtype=np.float32)
            y_array[list(range(len(labels))), label_index] = 1
            x_vector=self.word_index(contents)
            yield x_vector,y_array

    def batch_iter(self):
        for epoch in range(self.arg.num_epochs):
            # self.shuffle(self.arg.train_file)
            time_str = datetime.datetime.now().isoformat()
            print("{}: epoch {}".format(time_str,epoch ))
            for x_vector, y_array in self.gen_batch(self.arg.train_file,batch_size=self.arg.batch_size):
                yield x_vector, y_array


if __name__ == '__main__':
    arg=argument()
    log=log_config()
    data=Data(arg,log)
    for x_vector,y_array in data.gen_batch(arg.valid_file):
        # print(x_vector)
        pass



