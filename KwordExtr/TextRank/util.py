#-*- encoding:utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import math
import networkx as nx
import numpy as np
import sys

sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n']
#                   名形词, 成语,简称略语,习用语,名词,人名,名词方位语素,地名,机构团体,其他专名,时间词,动词,副动词,名动词,英文
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']

text_type    = str
string_types = (str,)

def as_text(v):  ## 生成unicode字符串
    if v is None:
        return None
    elif isinstance(v, bytes):
        return v.decode('utf-8', errors='ignore')
    elif isinstance(v, str):
        return v
    else:
        raise ValueError('Unknown type %r' % type(v))


class AttrDict(dict):
    """Dict that can get attribute by dot"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def combine(word_list, window = 2):
    """构造在window下的单词组合，用来构造单词之间的边。
    
    Keyword arguments:
    word_list  --  list of str, 由单词组成的列表。
    windows    --  int, 窗口大小。
    """
    if window < 2: window = 2
    for x in range(1, window):
        if x >= len(word_list):
            break
        word_list2 = word_list[x:]
        res = zip(word_list, word_list2)
        for r in res:
            yield r

def get_similarity(word_list1, word_list2):
    """默认的用于计算两个句子相似度的函数。

    Keyword arguments:
    word_list1, word_list2  --  分别代表两个句子，都是由单词组成的列表
    """
    words   = list(set(word_list1 + word_list2))        
    vector1 = [float(word_list1.count(word)) for word in words]
    vector2 = [float(word_list2.count(word)) for word in words]
    
    vector3 = [vector1[x]*vector2[x]  for x in range(len(vector1))]
    vector4 = [1 for num in vector3 if num > 0.]
    co_occur_num = sum(vector4)

    if abs(co_occur_num) <= 1e-12:
        return 0.
    
    denominator = math.log(float(len(word_list1))) + math.log(float(len(word_list2))) # 分母
    
    if abs(denominator) < 1e-12:
        return 0.
    
    return co_occur_num / denominator



def sort_words(vertex_source, edge_source, window = 2, pagerank_config = {'alpha': 0.85,}):
    """将单词按关键程度从大到小排序

    Keyword arguments:
    vertex_source   --  二维列表，子列表代表句子，子列表的元素是单词，这些单词用来构造pagerank中的节点
    edge_source     --  二维列表，子列表代表句子，子列表的元素是单词，根据单词位置关系构造pagerank中的边
    window          --  一个句子中相邻的window个单词，两两之间认为有边
    pagerank_config --  pagerank的设置
    """
    sorted_words   = []
    word_index     = {}
    index_word     = {}
    _vertex_source = vertex_source
    _edge_source   = edge_source
    words_number   = 0
    for word_list in _vertex_source:
        for word in word_list:
            if not word in word_index:
                word_index[word] = words_number
                index_word[words_number] = word
                words_number += 1

    graph = np.zeros((words_number, words_number))
    
    for word_list in _edge_source:
        for w1, w2 in combine(word_list, window):
            if w1 in word_index and w2 in word_index:
                index1 = word_index[w1]
                index2 = word_index[w2]
                graph[index1][index2] += 1.0
                graph[index2][index1] += 1.0

    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank(nx_graph, **pagerank_config)          # this is a dict
    sorted_scores = sorted(scores.items(), key = lambda item: item[1], reverse=True)
    for index, score in sorted_scores:
        item = AttrDict(word=index_word[index], weight=score)
        sorted_words.append(item)

    return sorted_words

def sort_sentences(sentences, words, sim_func = get_similarity, pagerank_config = {'alpha': 0.85,}):
    """将句子按照关键程度从大到小排序

    Keyword arguments:
    sentences         --  列表，元素是句子
    words             --  二维列表，子列表和sentences中的句子对应，子列表由单词组成
    sim_func          --  计算两个句子的相似性，参数是两个由单词组成的列表
    pagerank_config   --  pagerank的设置
    """
    sorted_sentences = []
    _source = words
    sentences_num = len(_source)        
    graph = np.zeros((sentences_num, sentences_num))
    
    for x in range(sentences_num):
        for y in range(x, sentences_num):
            similarity = sim_func( _source[x], _source[y] )
            graph[x, y] = similarity
            graph[y, x] = similarity
            
    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank(nx_graph, **pagerank_config)              # this is a dict
    sorted_scores = sorted(scores.items(), key = lambda item: item[1], reverse=True)

    for index, score in sorted_scores:
        item = AttrDict(index=index, sentence=sentences[index], weight=score)
        sorted_sentences.append(item)

    return sorted_sentences

"""
关键字提取算法的相关配置
"""
from stanfordcorenlp import StanfordCoreNLP
from nltk.tokenize import regexp_tokenize,word_tokenize
from nltk.tag import pos_tag
from nltk.stem import PorterStemmer,LancasterStemmer,RSLPStemmer,SnowballStemmer
import re,unicodedata
from contractions import contractions_dict

# 常用名词(单数),常用名词(复数),专有名词(单数),专有名词(复数),形容词或序数词,形容词比较级,形容词最高级,动词基本形式,动词过去式,动名词和现在分词,过去分词,动词非第三人称单数,动词第三人称单数
ALLOW_SPEECH_TAGS= ['nn', 'nns', 'nnp', 'nnps', 'jj', 'jjr', 'jjs', 'vb', 'vbd', 'vbg', 'vbn', 'vbp', 'vbz']

# ===============保存模型文件路径=======================
PARAMSFILE="./model/newparams.json"
LDAFILE="./model/LDA.model"
VECTORMODELFILE="./model/Vector.model"
LDAMODELFILE="./model/LdaModel.model"
STANF_MODULE=StanfordCoreNLP("./stanford-corenlp-full-2018-02-27")


def remove_accented_chars(text):
    """
    去除英文中的重音字符，如将ě转换成e
    :param text:
    :return:
    """
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def expand_contractions(text, contraction_mapping=contractions_dict):
    """
    扩展缩写，如，don't 转换成do not
    :param text:
    :param contraction_mapping:
    :return:
    """
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def as_text(v):
    ## 生成unicode字符串
    v=v.replace("\u2013",'-')
    v=v.replace("\u2014","--")
    table={ord(f):ord(t) for f,t in zip(u"""，‘’“”。！？【】（）％＃＠＆１２３４５６７８９０–""",u""",''"".!?[]()%#@&1234567890-""")}
    v=v.translate(table)
    v=remove_accented_chars(v)
    v=expand_contractions(v)
    if v is None:
        return None
    elif isinstance(v, bytes):
        return v.decode('utf-8', errors='ignore')
    elif isinstance(v, str):
        return v
    else:
        raise ValueError('Unknown type %r' % type(v))


def readStopFile(path=None):
    if path is None:
        path="./enStopwords.txt"
    with open(path,'r') as f:
        data=f.readlines()
    data=[da.strip() for da in data]
    return data


def is_number(chars):
    try:
        float(chars) if "." in chars else int(chars)
        return True
    except ValueError:
        return False


def filterStopWord(word_seq,stop_words=None,attr_dict=False):
    """
    去除停用词
    :param word_seq:
    :return:
    """
    if stop_words is None:
        stop_words=readStopFile()
    seg_words=[]
    for sent in word_seq:
        while isinstance(sent,list):
            seg_words.append(filterStopWord(sent,stop_words=stop_words))
            break
        if isinstance(sent,str):
            if sent not in stop_words:
                seg_words.append(sent)
        if isinstance(sent,AttrDict):
            if sent.word not in stop_words:
                seg_words.append(sent)
        if isinstance(sent,tuple):
            word,pos=sent
            if word not in stop_words:
                seg_words.append(AttrDict(word=word,tag=pos))

    return seg_words


def filterTagsWord(word_seq,allow_speech_tags=ALLOW_SPEECH_TAGS):
    """过滤指定词性"""
    filteredWord=[]
    for word in word_seq:
        while isinstance(word,list):
            filteredWord.append(filterTagsWord(word))
            break

        if isinstance(word,AttrDict):
            if word.tag.lower() not in allow_speech_tags:
                filteredWord.append(word)
    return filteredWord

class AttrDict(dict):
    def __init__(self,*args,**kwargs):
        super(AttrDict,self).__init__(*args,**kwargs)
        self.__dict__=self

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key]=value
        else:
            self[key]=value

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            return self[item]

class Segment():
    def __init__(self,stop_words=None):
        if stop_words:
            self.stopWords=stop_words
        else:
            self.stopWords=readStopFile()

    def segSentence(self,text,stanford=True):
        """
        比较详细的分句和分词，使用nltk中的sent_tokenize不是很完善，所以先用正则分词，在进行合并。
        :param sentence:
        :return:
        """
        text=as_text(text)
        reg=re.compile("[\s\t\n ]+")
        text=reg.sub(" ",text)

        reg_pattern="""(?x)                            # 设置以编写较长的正则条件
                    (?:(?:(?:[A-Z]\.)+[A-Z]?)(?:\w+))                     # 缩略词 
                    |\$?\d+(?:[,\.:-])(?:\d+)*%?           # 货币、百分数、比分、时间(H:m)、日期
                    |Mr\.|Mrs\.|Ms\.
                    |\w+(?:[-':]\w*)*                   # 用连字符链接的词汇
                    |\.\.\.                            # 省略符号 
                    |(?:[.;,"'?(){}:-_`])                # 特殊含义字符 
                    """

        if stanford:
            wordseg=STANF_MODULE.word_tokenize(text)
        else:
            wordseg=regexp_tokenize(text,reg_pattern)
        index=0
        sentences=[]
        seg_words=[]
        single_sent=[]
        for word in wordseg:
            index+=len(word)
            single_sent.append(word)
            if word in ['.',"?","!"]:
                sentences.append(" ".join(single_sent))
                seg_words.extend(single_sent)
                single_sent=[]
            if len(text)>index and text[index]==' ':
                index+=1
        if len(single_sent):
            sentences.append(" ".join(single_sent))
            seg_words.extend(single_sent)
        # 预防最后一句可能没有标点符号结尾
        return sentences,seg_words

    def segment(self,sentences,stem=False):
        """
        对文本进行词性分词,并对词语进行词性过滤和停用词过滤
        :param text:
        :return: 返回进行停用词和词性过滤的词组以及返回没有进行任何过滤的词组
        """
        if stem:
            porter_stemmer=PorterStemmer()
            # 对停用词提取词干
            self.stopWords=[porter_stemmer.stem(w) for w in self.stopWords]
        # 先进行分句，在进行分词
        words=[pos_tag(word_tokenize(s)) for s in sentences]
        no_filtered_words=[]#没有进行停用词和词性过滤
        for sent in words:
            newsent1=[]
            index=0
            for w,pos in sent:
                if stem:
                    # 提取词干
                    w=porter_stemmer.stem(w)
                newsent1.append(AttrDict(word=w.strip().lower(),tag=pos,index=index))
                index+=1
            no_filtered_words.append(newsent1)
        return no_filtered_words

    def filterWord(self,words,is_filter_stop=True,is_filter_tags=False):
        """
        :param words: 分好词的序列
        :param is_filter_stop: 是否过滤停用词
        :param is_filter_tags: 是否过滤词性
        :return:
        """
        # 过滤停用词
        if is_filter_stop:
            words=filterStopWord(words)

        # 按照词性过滤
        if is_filter_tags:
            words=filterTagsWord(words)

        return words



if __name__ == '__main__':
    pass