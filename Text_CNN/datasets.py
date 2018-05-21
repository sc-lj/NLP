# coding:utf-8

import zipfile
import chardet,codecs
from xml.dom import minidom
from urllib.parse import urlparse
from multiprocessing.dummy import Pool
import string,re
import numpy as np
from collections import Counter
from .config import *


# 根据搜狐新闻的网址的host，将这些划分如下
dicurl={'media':'传媒','baobao':'母婴','stock':'金融','it':'IT','fund':'金融','bschool':'商业','expo2010':"城市",'auto':"汽车",
        "cul":"文化","2012":"体育","goabroad":'留学','yule':"娱乐","travel":"旅游","2010":"体育","astro":"星座","sh":"城市",
        "women":"健康","s":"体育","dm":"动漫","chihe":"美食","2008":"体育","learning":"留学","business":"商业",
        "gongyi":"公益","men":"健康","health":"健康","sports":"体育","money":"金融","green":"美食","gd":"城市"}

filename='/Users/lj/Downloads/news_sohusite_xml.full.zip'
# 解压zip文件
def extract_zip(filename):
    with zipfile.ZipFile(filename) as files:
        for single in files.namelist():
            files.extract(single,'./')

#
def detext_souhu_encod(filename):
    f=open(filename,'rb')
    data=f.read()
    predict=chardet.detect(data)
    f.close()
    return predict['encoding']

def deal_souhu_corpus(data):
    pass

def read_souhu_corpus(filename,encode):
    dometree=[]
    with open(filename,'r',encoding=encode,errors='ignore') as f:
        for line in f:
            if '<doc>' in line:
                line='<docs>\n'
                dometree=[]
            dometree.append(line)
            if '</doc>' in line:
                lines= ''.join(dometree)
                lines=lines.replace('</doc>','</docs>')
                lines=lines.replace('&', '.')# 语料库中要将&替换成.，不然会出错的
                yield lines.encode('utf-8')

def map_function(da):
    doc=minidom.parseString(da)
    root=doc.documentElement
    url=root.getElementsByTagName('url')[0].childNodes[0].data
    host=urlparse(url).netloc
    host=host.split('.')[-3]
    try:
        content=root.getElementsByTagName('content')[0].childNodes[0].data
    except:
        content=''
    try:
        title=root.getElementsByTagName('contenttitle')[0].childNodes[0].data
    except:
        title=''
    return host,content,title




def write_file(data):
    pool=Pool(3)
    host_content=pool.map(map_function,data)
    pool.close()
    pool.join()
    with open('./new_sohu.txt','w') as f:
        for label,content,title in host_content:
            if label not in dicurl:
                continue
            lable=dicurl[label]
            f.write(lable+"++"+title+"++"+content.strip()+'\n')

# data=read_souhu_corpus('./news_sohusite_xml.dat','gb2312')
# write_file(data)


class DealData():
    def __init__(self,filename):
        self.FLAGS=seq_param()
        self.filename=filename
        self.vocab=set()#词汇表
        self.cont_label = []# 文本内容
        self.word_freq=Counter()#词频表,Counter能对key进行累加
        self. punctuation = re.compile(u"[-~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】,、；‘：“”，。、《》？「『」』\t\n]+")
        self.rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")  # 去除所有半角全角符号，只留字母、数字、中文。
        self.num=re.compile('\d{1,}')#将文本中的数字替换成该标示符
        # self.date=re.compile("((^((1[8-9]\d{2})|([2-9]\d{3}))([-\/\._])(10|12|0?[13578])([-\/\._])(3[01]|[12][0-9]|0?[1-9])$)|(^((1[8-9]\d{2})|([2-9]\d{3}))([-\/\._])(11|0?[469])([-\/\._])(30|[12][0-9]|0?[1-9])$)|(^((1[8-9]\d{2})|([2-9]\d{3}))([-\/\._])(0?2)([-\/\._])(2[0-8]|1[0-9]|0?[1-9])$)|(^([2468][048]00)([-\/\._])(0?2)([-\/\._])(29)$)|(^([3579][26]00)([-\/\._])(0?2)([-\/\._])(29)$)|(^([1][89][0][48])([-\/\._])(0?2)([-\/\._])(29)$)|(^([2-9][0-9][0][48])([-\/\._])(0?2)([-\/\._])(29)$)|(^([1][89][2468][048])([-\/\._])(0?2)([-\/\._])(29)$)|(^([2-9][0-9][2468][048])([-\/\._])(0?2)([-\/\._])(29)$)|(^([1][89][13579][26])([-\/\._])(0?2)([-\/\._])(29)$)|(^([2-9][0-9][13579][26])([-\/\._])(0?2)([-\/\._])(29)$))|(^\d{4}年\d{1,2}月\d{1,2}日$)$")
        self.date=re.compile("^(^(\d{4}|\d{2})(\-|\/|\.)\d{1,2}(\-|\/|\.)\d{1,2}$)|(^\d{4}年\d{1,2}月\d{1,2}日$)$")
        self.times=re.compile('\d{1,2}:\d{1,2}|[\d{1,2}点]*[\d{1,2}分]*')
        self.alpha=re.compile(string.ascii_letters)

    def read_file(self):
        with open(self.filename,'r') as f:
            data=f.readline(100)
            while data:
                for da in data:
                    host, content, title=da.split('++',2)
                    yield host,content,title
                data = f.readline(100)

    def get_label_content(self):
        for label, content, title in self.read_file():
            line=self.get_vocab(content)
            self.cont_label.append([line,label])

    def get_vocab(self,line):
        """
        处理文本，主要是剔除掉一些无意义的字符，并且提取出日期格式用<DATE>字符代替，数字用<NUM>字符代替
        :param line: 输入的是单个文本
        :return:
        """
        one_line=[]
        # 先替换日期或者数字、字母
        line=self.date.sub('\s<DATE>\s',line)#注意<DATE>前后要留空格，方便后面好分割
        line=self.num.sub('\s<NUM>\s',line)
        line=self.times.sub('',line)
        line=self.alpha.sub('',line)# 去掉字母
        line=self.punctuation.sub('',line)
        line=self.rule.sub('',line)
        lines=line.split('\s')
        for one in lines:
            if one.strip() in ['<DATE>','<NUM>','<TIMES>']:
                one_line.append(one)
            else:
                one_line.extend(list(one))
        self.vocab.update(set(one_line))
        self.word_freq.update(Counter(one_line))
        return one_line

    def gene_dict(self):
        """
        生成数字和字的映射
        :return:
        """
        self.word_id=dict(zip(self.vocab,range(len(self.vocab))))
        self.id_word=dict(zip(range(len(self.vocab)),self.vocab))

    def seq_vector(self,line):
        """
        生成seq-CNN模型需要的向量
        :param line: 单个文本的字串
        :return:
        """
        assert isinstance(line, list)
        text_vector=np.zeros([len(line),len(self.vocab)],dtype=np.int32)
        j=0
        # 将没有在词汇表中的字用零代替
        for word in line:
            if word in self.vocab:
                index=self.word_id[word]
                text_vector[j][index] = 1
            j+=1
        return text_vector

    def bow_vector(self,line,num=3):
        """
        生成bow-CNN模型需要的向量
        :param line: 单个文本的字串,是list形式
        :param num: 每个向量包含相邻的字，默认为3
        :return:
        """
        assert isinstance(line,list)
        line_len=len(line)
        text_vector=np.zeros([line_len-num+1,len(self.vocab)],dtype=np.int32)
        # 将没有在词汇表中的字用零代替
        for i in range(0,line_len-num):
            for word in line[i:i+num]:
                if word in self.vocab:
                    index=self.word_id[word]
                    text_vector[i][index]=1
        return text_vector

    def batch_iter(self,bow_seq='seq', shuffle=True):
        data_size=len(self.cont_label)
        data=np.array(self.cont_label)

        for epoch in range(self.FLAGS.num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffle_data = data[list(shuffle_indices)]
            else:
                shuffle_data = data

            for cont,label in shuffle_data:
                if bow_seq=='seq':
                    vector=self.seq_vector(cont)
                else:
                    vector=self.bow_vector(cont)
                yield vector,label




