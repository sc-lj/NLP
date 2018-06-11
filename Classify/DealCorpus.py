# coding:utf-8

import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import cpu_count,Pool,Process
from queue import Queue
from Config import *

import matplotlib.pyplot as plt
import zipfile,threading
import chardet,codecs
from xml.dom import minidom
from urllib.parse import urlparse
import string,re,time,json,os


# 根据搜狐新闻的网址的host，将这些划分如下
dicurl={'media':'传媒','baobao':'母婴','stock':'金融','it':'IT','fund':'金融','bschool':'商业','expo2010':"城市",'auto':"汽车",
        "cul":"娱乐文化","2012":"体育","goabroad":'教育','yule':"娱乐文化","travel":"旅游","2010":"体育","astro":"星座","sh":"城市",
        "women":"健康","s":"体育","dm":"动漫","chihe":"美食","2008":"体育","learning":"教育","business":"商业",
        "gongyi":"公益","men":"健康","health":"健康","sports":"体育","money":"金融","green":"美食","gd":"城市"}

# 解压zip文件
def extract_zip(filename):
    with zipfile.ZipFile(filename) as files:
        for single in files.namelist():
            files.extract(single,'./')

# 检测文本的编码格式
def detext_souhu_encod(filename):
    f=open(filename,'rb')
    data=f.read()
    predict=chardet.detect(data)
    f.close()
    return predict['encoding']

#
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
    try:
        content=root.getElementsByTagName('content')[0].childNodes[0].data
    except:
        content=''
    try:
        title=root.getElementsByTagName('contenttitle')[0].childNodes[0].data
    except:
        title=''
    return host,content,title

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def write_file(data):
    pool=Pool(3)
    host_content=pool.map(map_function,data)
    pool.close()
    pool.join()
    with open('./new_sohu_utf_8.txt','a+') as f:
        for host,content,title in host_content:
            content=content.replace('\n','')
            for label in host.split('.'):
                if label in dicurl:
                    labels=dicurl[label]
                    break
                else:
                    labels=None
            if labels==None:
                continue

            cont=labels+"++"+title+"++"+content.strip()
            cont=strQ2B(cont)
            f.write(cont+'\n')

def read_dir(dir):
    for files in os.listdir(dir):
        path=os.path.join(dir,files)
        data=read_souhu_corpus(path,'utf-8')
        write_file(data)

# data=read_souhu_corpus('/Users/apple/Downloads/SogouCA/news.allsites.sports.6307.txt','GB2312')
# write_file(data)

read_dir('../Dataset/corpus/SogouCA')


class Deal(object):
    queue = Queue()
    def __init__(self,arg,logger):
        self.punctuation = re.compile(u"[~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】,、；‘：“”，。、《》？「『」』＃\t\n]+")
        self.arg=arg
        self.filename=self.arg.corpus_txt
        self.target_file=self.arg.target_file
        self.stopfile=self.arg.stopfile
        self.logger=logger

        self.label_num={}

        self.multi_thread()


    def read_file(self):
        with open(self.filename,'r') as f:
            data=f.readline()
            while data:
                label,title, content=data.split('++',2)
                if len(content.strip()) == 0:
                    data = f.readline()
                    continue
                yield label,content,title
                data = f.readline()


    def multi_thread(self):
        self.logger.info('开始处理文本内容和标签')
        t1=threading.Thread(target=self.get_label_content)
        t1.start()
        time.sleep(3)
        t=threading.Thread(target=self.callback)
        t.start()

        ts=[]
        ts.extend([t1,t])
        for i in ts:
            i.join()
        print('语料库的文本和标签处理完毕')
        self.logger.info('语料库的文本和标签处理完毕')


    def readstopword(self):
        with open(self.stopfile,'r') as f:
            stopwords=f.read()
        return set(stopwords)


    def callback(self):
        f=open(self.target_file,'w',encoding='utf-8')
        j=0
        while True:
            if Deal.queue.empty():
                time.sleep(10)
                if Deal.queue.empty():
                    break
            one_line,label,title=Deal.queue.get()
            if len(one_line)<10:
                continue
            if label in self.label_num:
                self.label_num[label]+=1
            else:
                self.label_num[label]=1
            data=json.dumps({label+title:list(one_line)})
            f.write(data+'\n')
            j+=1
        f.close()
        print('had write num', j)
        print('write target file is end')


    def get_label_content(self):
        """
        利用多进程方法快速处理文本
        """
        stopword=self.readstopword()
        pool_num=cpu_count()-2
        # pool = Pool(processes=pool_num)
        # results=[]
        for label, content, title in self.read_file():
            if len(content.strip())==0:
                continue
            self.get_vocab(self,content,label,stopword,title)
            # pool.apply_async(self.get_vocab,(content,label,stopword,))
            # results.append(line_label)

        # pool.close()
        # pool.join()


    # 在类中使用multiprocessing，当multiprocessing需要执行类方法的时候，必须将该类方法装饰成静态方法。
    # 静态方法是无法调用实例属性的，所以需要将值当作参数。
    @staticmethod
    def get_vocab(self,line,label,stopword,title):
        """
        处理文本，主要是剔除掉一些无意义的字符，并且提取出日期格式用<DATE>字符代替，数字用<NUM>字符代替
        :param line: 输入的是单个文本
        :return:
        """
        print(self.punctuation)
        rule = re.compile(r"[^-a-zA-Z0-9\u4e00-\u9fa5]")  # 去除所有全角符号，只留字母、数字、中文。要保留-符号，以防2014-3-23时间类型出现,而被删除
        num=re.compile('\d{1,}')#将文本中的数字替换成该标示符
        # date=re.compile("((^((1[8-9]\d{2})|([2-9]\d{3}))([-\/\._])(10|12|0?[13578])([-\/\._])(3[01]|[12][0-9]|0?[1-9])$)|(^((1[8-9]\d{2})|([2-9]\d{3}))([-\/\._])(11|0?[469])([-\/\._])(30|[12][0-9]|0?[1-9])$)|(^((1[8-9]\d{2})|([2-9]\d{3}))([-\/\._])(0?2)([-\/\._])(2[0-8]|1[0-9]|0?[1-9])$)|(^([2468][048]00)([-\/\._])(0?2)([-\/\._])(29)$)|(^([3579][26]00)([-\/\._])(0?2)([-\/\._])(29)$)|(^([1][89][0][48])([-\/\._])(0?2)([-\/\._])(29)$)|(^([2-9][0-9][0][48])([-\/\._])(0?2)([-\/\._])(29)$)|(^([1][89][2468][048])([-\/\._])(0?2)([-\/\._])(29)$)|(^([2-9][0-9][2468][048])([-\/\._])(0?2)([-\/\._])(29)$)|(^([1][89][13579][26])([-\/\._])(0?2)([-\/\._])(29)$)|(^([2-9][0-9][13579][26])([-\/\._])(0?2)([-\/\._])(29)$))|(^\d{4}年\d{1,2}月\d{1,2}日$)$")
        date=re.compile("((\d{4}|\d{2})(\-|\/|\.)\d{1,2}(\-|\/|\.)\d{1,2})|((\d{4}年)?\d{1,2}月\d{1,2}日)")
        times=re.compile('(\d{1,2}:\d{1,2})|((\d{1,2}点\d{1,2}分)|(\d{1,2}时))')
        alpha=re.compile(string.ascii_letters)
        one_line = []
        line=alpha.sub('',line)# 去掉字母
        line=self.punctuation.sub('',line)#

        line = date.sub(' <DATE> ', line)  # 注意<DATE>前后要留空格，方便后面好分割
        line = num.sub(' <NUM> ', line)
        line = times.sub(' <DATE> ', line)
        line = re.sub('-','',line)
        line = line.split(' ')
        for one in line:
            if one.strip() in ['<DATE>','<NUM>','<TIMES>']:
                one_line.append(one)
            else:
                one_line.extend(list(one))

        one_line=self.diff(one_line,stopword)
        if Deal.queue.full():
            time.sleep(0.5)
        if len(one_line)>10:
            Deal.queue.put([one_line,label,title])
        # return label,list(one_line)

    def diff(self,words,stopwords):
        """
        去掉words中包含的停用词，如果用set().difference()会导致得到的词序改变
        :param words: 文本词汇
        :param stopwords: 停用词
        :return:
        """
        newwords=[]
        for word in words:
            if word not in stopwords:
                newwords.append(word)
        return newwords


# if __name__ == '__main__':
#     arg=argument()
#     logger=log_config()
#     deal=Deal(arg,logger)


