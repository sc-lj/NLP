# coding:utf-8

import tensorflow as tf
from collections import defaultdict

FLAGS=tf.flags.FLAGS



class Vocab():
    def __init__(self,corpus_file=None):
        self.corpus=defaultdict(int)
        self.words_to_id={}
        self.id_to_word={}
        self.word_count=0
        self.UNK =u'UNK'


        if not corpus_file:
            self._genCorpus()
        f=open(FLAGS.corpus_file,'r')
        lines=f.readlines()
        for i,line in enumerate(lines):
            piece=line.split()
            if len(piece)!=2:
                raise ValueError("文件{0}第{1}行数据{2}错误,".format(corpus_file,i,line))
            self.words_to_id[piece[0]]=self.word_count
            self.id_to_word[self.word_count]=piece[0].strip()
            self.word_count+=1
        if self.UNK not in self.words_to_id:
            self.words_to_id[self.UNK]=self.word_count
            self.id_to_word[self.word_count]=self.UNK
            self.word_count+=1

    def _genCorpus(self):
        f = open(FLAGS.source_file, 'r')
        lines = f.readlines()
        for line in lines:
            pieces = line.split()
            for piece in pieces:
                self.corpus[piece.strip()] += 1

        corpus_file = "../data/corpus_file.txt"
        f = open(corpus_file, 'w')
        for k, v in self.corpus.items():
            f.write(k + " " + str(v) + "\n")
        f.close()

        FLAGS.corpus_file = corpus_file

    def checkWord(self,word):
        if word not in self.words_to_id:
            return None
        return self.words_to_id[word]

    def wordToId(self,word):
        if word not in self.words_to_id:
            return self.words_to_id[self.UNK]
        return self.words_to_id[word]

    def idToWord(self,id):
        if id not in self.id_to_word:
            raise ValueError("id 不在voab中:{}".format(id))
        return self.id_to_word[id]

    def NumIds(self):
        return self.word_count


def GetWordIds(vocab,text):
    ids=[]
    for w in text:
        i=vocab.wordToId(w)
        ids.append(i)
    return ids,


if __name__ == '__main__':
    vocab=Vocab('../data/xinhua_source.txt')