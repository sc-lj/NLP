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

import Config
import json

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


    def read_corpus(self, filename, batch_size=1):
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


    def gen_batch(self):
        for labels, contents in self.read_corpus(''):

            pass


