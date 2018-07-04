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
import time

class Data():
    def __init__(self, arg, logger, max_sequence_length=None):
        self.arg = arg
        self.logger = logger
        if max_sequence_length:
            self.max_sequence_length = max_sequence_length
        else:
            self.max_sequence_length = arg.max_sequence_length
        self.labels = ['汽车', '商业', '健康', '体育', 'IT', '旅游', '娱乐文化', '教育']
        self.gene_dict()

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


    def read_corpus(self, filename, batch_size=2):
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

    def gen_batch(self,filename):
        for labels, contents in self.read_corpus(filename):
            label_index = list(map(self.labels.index, labels))
            y_array = np.zeros([len(labels), len(self.labels)], dtype=np.float32)
            y_array[list(range(len(labels))), label_index] = 1
            contents_index=[]
            print(contents)
            # for con in contents:
            #     if con in self.vocab:
            #         pass





if __name__ == '__main__':
    arg=argument()
    log=log_config()
    data=Data(arg,log)
    data.gen_batch(arg.valid_file)




