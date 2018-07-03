# coding:utf-8

import os,sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import cpu_count,Pool,Process,Queue,Value,Manager
import multiprocessing as mu
from GloConfi import *
import matplotlib.pyplot as plt
import zipfile,threading
import chardet,codecs
from xml.dom import minidom
from urllib.parse import urlparse
import string,re,time,json,os
from itertools import groupby
from collections import Counter,defaultdict

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
    with open('../Dataset/new_sohu.txt','a+') as f:
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
        data=read_souhu_corpus(path,'gb2312')
        write_file(data)

# 对queue数量进行限制，不然其将占满整个内存空间
# 特别是对于读取不平衡的情况
queues = Queue()

class Deal(object):
    def __init__(self,arg,logger):
        self.arg=arg
        self.logger=logger

    def read_file(self):
        with open(self.arg.corpus_txt,'r',encoding="utf-8") as f:
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
        t1=Process(target=self.get_label_content,args=(queues,))
        t1.start()
        time.sleep(3)
        value=Value('d',0)
        f = open(self.arg.target_file, 'a+', encoding='utf-8')
        t2=Process(target=self.callback,args=(queues,f,value),name='t1')
        t2.start()
        t3 = Process(target=self.callback, args=(queues,f,value),name='t3')
        t3.start()
        ts=[]
        ts.extend([t1,t2,t3])
        for i in ts:
            i.join()
        f.close()
        self.logger.info(u'语料库的文本和标签处理完毕')


    def readstopword(self):
        with open(self.arg.stopfile,'r',encoding="utf-8") as f:
            stopwords=f.read()
        return set(stopwords)


    def callback(self,q,f,j):
        """
        :param q: 队列
        :param f: 文件对象
        :param j: 进程间共享变量
        :return:
        """
        while q.qsize():
            data=q.get()
            if data is None:
                break
            f.write(data+'\n')
            j.value+=1
            if j.value%1000==0:
                print(mu.current_process().name,'A'*7,j.value,q.qsize())
        """
        join_thread()
        调用close()方法后可以调用join_thread()方法保证缓冲数据一定会被刷新到管道。
        默认情况下，如果该进程不是队列的创建者，会自动调用此方法。可以调用cancel_join_thread()使该方法失效。

        cancel_join_thread()
        取消join_thread()的作用，不过一般情况下不会使用，因为它会容易造成数据丢失。
        """
        # q.cancel_join_thread()
        print('had write num', j.value,mu.current_process().name)
        print('write target file is end')


    def get_label_content(self,q):
        """
        利用多进程方法快速处理文本
        """
        stopword=self.readstopword()
        for label, content, title in self.read_file():
            if len(content.strip())==0:
                continue
            self.get_vocab(self,content,label,stopword,title,q)
        # 生产者需要告知消费者没有更多项目了，消费者可以关闭了。这时需要传递给消费者某个信号，告诉消费者没有更多项目了，可以关闭了。
        # 注意：每个消费者都需要一个信号，所以有多少个消费者，就需要多少个信号。
        for i in range(2):
            q.put(None)


    # 在类中使用multiprocessing，当multiprocessing需要执行类方法的时候，必须将该类方法装饰成静态方法。
    # 静态方法是无法调用实例属性的，所以需要将值当作参数。
    @staticmethod
    def get_vocab(self,line,label,stopword,title,q):
        """
        处理文本，主要是剔除掉一些无意义的字符，并且提取出日期格式用<DATE>字符代替，数字用<NUM>字符代替
        :param line: 输入的是单个文本
        :return:
        """
        punctuation = re.compile(u"[~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】,、；‘：“”，。、《》？「『」』＃\t\n∶┛↓┑┏π]+")
        rule = re.compile(r"[^' '<>a-zA-Z0-9\u4e00-\u9fa5]")  # 去除所有全角符号，只留字母、数字、中文。要保留-符号，以防2014-3-23时间类型出现,而被删除
        num=re.compile('\d{1,}')#将文本中的数字替换成该标示符
        # date=re.compile("((^((1[8-9]\d{2})|([2-9]\d{3}))([-\/\._])(10|12|0?[13578])([-\/\._])(3[01]|[12][0-9]|0?[1-9])$)|(^((1[8-9]\d{2})|([2-9]\d{3}))([-\/\._])(11|0?[469])([-\/\._])(30|[12][0-9]|0?[1-9])$)|(^((1[8-9]\d{2})|([2-9]\d{3}))([-\/\._])(0?2)([-\/\._])(2[0-8]|1[0-9]|0?[1-9])$)|(^([2468][048]00)([-\/\._])(0?2)([-\/\._])(29)$)|(^([3579][26]00)([-\/\._])(0?2)([-\/\._])(29)$)|(^([1][89][0][48])([-\/\._])(0?2)([-\/\._])(29)$)|(^([2-9][0-9][0][48])([-\/\._])(0?2)([-\/\._])(29)$)|(^([1][89][2468][048])([-\/\._])(0?2)([-\/\._])(29)$)|(^([2-9][0-9][2468][048])([-\/\._])(0?2)([-\/\._])(29)$)|(^([1][89][13579][26])([-\/\._])(0?2)([-\/\._])(29)$)|(^([2-9][0-9][13579][26])([-\/\._])(0?2)([-\/\._])(29)$))|(^\d{4}年\d{1,2}月\d{1,2}日$)$")
        date=re.compile("((\d{4}|\d{2})(\-|\/|\.)\d{1,2}(\-|\/|\.)\d{1,2})|((\d{4}年)?\d{1,2}月\d{1,2}日)")
        times=re.compile('(\d{1,2}:\d{1,2})|((\d{1,2}点\d{1,2}分)|(\d{1,2}时))')
        alpha=re.compile(r"[a-z0-9]")
        one_line = []

        line=line.lower()
        line=re.sub("[<>' ']",'',line)
        line = date.sub(' <DATE> ', line)  # 注意<DATE>前后要留空格，方便后面好分割
        line = num.sub(' <NUM> ', line)
        line = times.sub(' <DATE> ', line)
        line = rule.sub('', line)  #
        line = alpha.sub('', line)  # 去掉字母
        line = re.sub("[-]",'',line)
        line = line.split(' ')
        for one in line:
            if one.strip() in ['<DATE>','<NUM>','<TIMES>']:
                one_line.append(one)
            else:
                one_line.extend(list(one))

        one_line=self.diff(one_line,stopword)
        if q.full():
            time.sleep(1)
        if len(one_line)>50:
            data = json.dumps({"label": label, "title": title, "content": list(one_line)})
            q.put(data)

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

    def read_corpus(self):
        with open(self.arg.target_file,'r',encoding='utf-8') as f:
            data=f.readline()
            while data:
                jsdata=json.loads(data)
                label,one_line=jsdata['label'],jsdata['content']
                yield label,data,one_line
                data = f.readline()

    def group(self,num,grouplist=[50,200,400]):
        for i in grouplist:
            if num<=i:
                return str(i)
        return '>%s'%grouplist[-1]


    def analysis_corpus(self):
        label_num = defaultdict(int)
        content_len=[]
        for label, data,one_line in self.read_corpus():
            label_num[label]+=1
            content_len.append(len(one_line))

        content_len=sorted(content_len)

        newgroup=[(k,list(g)) for k,g in groupby(content_len,key=self.group)]
        newgroup=[(k,len(g)) for k,g in newgroup]
        newgroup=dict(newgroup)
        print(newgroup)

        # 设置图形大小:figsize=(10, 6)
        plt.figure()
        #
        #通过栅格的形式创建布局方式,(2,1)创建2x1的布局形式，(0,0)绘制的位置，0行0列的位置绘制 ;colspan:表示跨几列 rowspan:表示跨几行
        ax1=plt.subplot2grid([2,1],[0,0])
        X=list(label_num.keys())
        Y=list(label_num.values())
        # width柱状图宽度，
        ax1.bar(X, Y, width=0.9, align='center', color='blue', alpha=0.8)
        # rotation是倾斜30度
        # plt.xticks(X, rotation=30)
        # 设置数字标签
        for a, b in zip(X, Y):
            plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=13)

        plt.subplot(2, 1, 2)
        X = list(newgroup.keys())
        Y = list(newgroup.values())
        # width柱状图宽度，
        plt.bar(X, Y, width=0.9, align='center', color='blue', alpha=0.8)
        # rotation是倾斜30度
        # plt.xticks(X, rotation=30)
        # 设置数字标签
        for a, b in zip(X, Y):
            plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=13)

        plt.show()

    def writefile(self,filename,data):
        with open(filename,'a+',encoding='utf-8') as f:
            if isinstance(data,str):
                f.write(data)
            elif isinstance(data,set) or isinstance(data,list):
                for da in data:
                    f.write(da+"\n")
            elif isinstance(data,dict):
                for key,value in data.items():
                    f.write(key+" "+str(value)+"\n")

    def gen_test_train_corpus(self):
        """
        生成测试和训练数据集
        """
        threshold=5000
        lables=defaultdict(int)
        vocab=defaultdict(int)
        for label, data,one_line in self.read_corpus():
            if lables[label]<=threshold:
                self.writefile(self.arg.valid_file,data)
            else:
                self.writefile(self.arg.train_file,data)
            lables[label]+=1
            for one in one_line:
                vocab[one]+=1
        newvocab={}
        vocab=dict(vocab)
        for key,value in vocab.items():
            if value>1000:
                newvocab[key]=value

        self.writefile(self.arg.vocab_file,newvocab)

if __name__ == '__main__':
    # data=read_souhu_corpus('/Users/apple/Downloads/SogouCA/news.allsites.sports.6307.txt','GB2312')
    # write_file(data)
    # read_dir('/Users/apple/Downloads/SogouCS')
    arg=argument()
    logger=log_config()
    deal=Deal(arg,logger)
    deal.multi_thread()
    # deal.analysis_corpus()
    deal.gen_test_train_corpus()


