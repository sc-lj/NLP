# coding:utf-8

import zipfile
import chardet,codecs
from xml.dom import minidom
from urllib.parse import urlparse
from multiprocessing.dummy import Pool
import string,re,time
import numpy as np
from collections import Counter


# 根据搜狐新闻的网址的host，将这些划分如下
dicurl={'media':'传媒','baobao':'母婴','stock':'金融','it':'IT','fund':'金融','bschool':'商业','expo2010':"城市",'auto':"汽车",
        "cul":"文化","2012":"体育","goabroad":'留学','yule':"娱乐","travel":"旅游","2010":"体育","astro":"星座","sh":"城市",
        "women":"健康","s":"体育","dm":"动漫","chihe":"美食","2008":"体育","learning":"留学","business":"商业",
        "gongyi":"公益","men":"健康","health":"健康","sports":"体育","money":"金融","green":"美食","gd":"城市"}

filename='./news_sohusite_xml.full.zip'
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
            content=content.replace('\n','')
            if label not in dicurl:
                continue
            lable=dicurl[label]
            f.write(lable+"++"+title+"++"+content.strip()+'\n')

# data=read_souhu_corpus('../datasets/news_sohusite_xml.dat','gb2312')
# write_file(data)


class DealData():
    def __init__(self,filename,FLAGS):
        self.FLAGS=FLAGS
        self.filename=filename
        self.max_sequence_length = 0

        self.labels=set()# 标签集
        self.vocab=set()#词汇表
        self.cont_label = []# 文本内容和标签
        self.word_freq=Counter()#词频表,Counter能对key进行累加

        self. punctuation = re.compile(u"[-~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】,、；‘：“”，。、《》？「『」』\t\n]+")
        self.rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")  # 去除所有半角全角符号，只留字母、数字、中文。
        self.num=re.compile('\d{1,}')#将文本中的数字替换成该标示符
        # self.date=re.compile("((^((1[8-9]\d{2})|([2-9]\d{3}))([-\/\._])(10|12|0?[13578])([-\/\._])(3[01]|[12][0-9]|0?[1-9])$)|(^((1[8-9]\d{2})|([2-9]\d{3}))([-\/\._])(11|0?[469])([-\/\._])(30|[12][0-9]|0?[1-9])$)|(^((1[8-9]\d{2})|([2-9]\d{3}))([-\/\._])(0?2)([-\/\._])(2[0-8]|1[0-9]|0?[1-9])$)|(^([2468][048]00)([-\/\._])(0?2)([-\/\._])(29)$)|(^([3579][26]00)([-\/\._])(0?2)([-\/\._])(29)$)|(^([1][89][0][48])([-\/\._])(0?2)([-\/\._])(29)$)|(^([2-9][0-9][0][48])([-\/\._])(0?2)([-\/\._])(29)$)|(^([1][89][2468][048])([-\/\._])(0?2)([-\/\._])(29)$)|(^([2-9][0-9][2468][048])([-\/\._])(0?2)([-\/\._])(29)$)|(^([1][89][13579][26])([-\/\._])(0?2)([-\/\._])(29)$)|(^([2-9][0-9][13579][26])([-\/\._])(0?2)([-\/\._])(29)$))|(^\d{4}年\d{1,2}月\d{1,2}日$)$")
        self.date=re.compile("^(^(\d{4}|\d{2})(\-|\/|\.)\d{1,2}(\-|\/|\.)\d{1,2}$)|(^\d{4}年\d{1,2}月\d{1,2}日$)$")
        self.times=re.compile('\d{1,2}:\d{1,2}|[\d{1,2}点]*[\d{1,2}分]*')
        self.alpha=re.compile(string.ascii_letters)

        self.get_label_content()
        self.gene_dict()

        # 词向量的长度
        self.vector_length=len(self.vocab)
        self.lable_length=len(self.labels)

    def read_file(self):
        with open(self.filename,'r') as f:
            data=f.readline()
            while data:
                host, content, title=data.split('++',2)
                yield host,content,title
                data = f.readline()

    def get_label_content(self):
        for label, content, title in self.read_file():
            self.labels.add(label)
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
        if len(one_line)>self.max_sequence_length:
            self.max_sequence_length=len(one_line)
        return one_line

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
        text_vector=np.zeros([self.max_sequence_length,len(self.vocab)],dtype=np.float32)
        j=0
        # 将没有在词汇表中的字用零代替
        for word in line:
            if word in self.vocab:
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
        if len(lines.shape)==1:
            text_vectors=self.single_seq_vector(lines)
        elif len(lines.shape)==2:
            text_vectors=[]
            text_vectors.append(self.single_seq_vector(line) for line in lines)
        else:
            raise('要生成seq模型的文本输入的维度不在处理范围内')
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

        if len(lines.shape)==1:
            text_vectors=self.single_bow_line(lines,num)
        elif len(lines.shape)==2:
            text_vectors=[]
            for line in lines:
                text_vector=self.single_bow_line(line,num)
                text_vectors.append(text_vector)
        else:
            raise('要生成bow模型的文本输入的维度不在处理范围内')
        return text_vectors

    def slice_batch(self,bow_seq='seq'):
        labellist=list(self.labels)
        data_size=len(self.cont_label)
        data=np.array(self.cont_label)
        X,Y =np.transpose(data)

        # 验证集的大小
        dev_data_size = -1 * int(self.FLAGS.dev_sample_percent * data_size)
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
        for epoch in range(self.FLAGS.num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                x_train = x_train[list(shuffle_indices)]
                y_train = y_train[list(shuffle_indices)]

            for num in range(0,data_size,self.FLAGS.batch_size):
                end_index=num+self.FLAGS.batch_size
                if data_size> end_index:
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
                yield (x_train_vector,y_train_array)

