# coding:utf-8

import threading
import string,re,time,json,os
from collections import Counter
from multiprocessing import cpu_count,Pool,Process
from queue import Queue
from Config import *

def callbackfun():
    pass

class Deal(object):
    queue = Queue(5000)
    def __init__(self,arg,logger):
        self.arg=arg
        self.filename=self.arg.corpus_txt
        self.stopfile=self.arg.stopfile
        self.logger=logger
        self.max_sequence_length = 0

        self.labels=set()# 标签集
        self.vocab=set()#词汇表
        self.cont_label = []# 文本内容和标签
        self.word_freq=Counter()#词频表,Counter能对key进行累加

        self.get_label_content()

        # 词向量的长度
        self.vector_length=len(self.vocab)
        self.lable_length=len(self.labels)

    def read_file(self):
        with open(self.filename,'r') as f:
            data=f.readline()
            while data:
                label,title, content=data.split('++',2)
                yield label,content,title
                data = f.readline()

    def args(self):
        return self.arg.target_file

    @staticmethod
    def callback(self,label,one_line):
        print(label, len(one_line))
        with open(self.arg.target_file,'w') as f:
            data=json.dumps({label:one_line})
            f.writelines(data+'\n')

    def multi_thread(self):
        self.logger.info('开始处理文本内容和标签')
        t1=threading.Thread(target=self.get_label_content)
        t1.start()
        time.sleep(3)
        t=threading.Thread(target=self.callback)
        t.start()

        t1.join()
        t.join()
        self.logger.info('语料库的文本和标签处理完毕')

    def get_label_content(self):
        """
        利用多进程方法快速处理文本
        """

        stopword=self.readstopword()
        pool_num=cpu_count()-2
        pool = Pool(processes=pool_num)
        # results=[]
        for label, content, title in self.read_file():
            if len(content.strip())==0:
                continue
            # self.get_vocab(content,label,stopword)
            pool.apply_async(self.get_vocab,args=(self,content,label,stopword,),callback=self.callback)
            # results.append(line_label)

        pool.close()
        pool.join()

    def readstopword(self):
        with open(self.stopfile,'r') as f:
            stopwords=f.read()
        return set(stopwords)

    # 在类中使用multiprocessing，当multiprocessing需要执行类方法的时候，必须将该类方法装饰成静态方法。
    # 静态方法是无法调用实例属性的，所以需要将值当作参数。
    @staticmethod
    def get_vocab(self,line,label,stopword):
        """
        处理文本，主要是剔除掉一些无意义的字符，并且提取出日期格式用<DATE>字符代替，数字用<NUM>字符代替
        :param line: 输入的是单个文本
        :return:
        """
        punctuation = re.compile(u"[~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】,、；‘：“”，。、《》？「『」』＃\t\n]+")
        rule = re.compile(r"[^-a-zA-Z0-9\u4e00-\u9fa5]")  # 去除所有全角符号，只留字母、数字、中文。要保留-符号，以防2014-3-23时间类型出现
        num=re.compile('\d{1,}')#将文本中的数字替换成该标示符
        # date=re.compile("((^((1[8-9]\d{2})|([2-9]\d{3}))([-\/\._])(10|12|0?[13578])([-\/\._])(3[01]|[12][0-9]|0?[1-9])$)|(^((1[8-9]\d{2})|([2-9]\d{3}))([-\/\._])(11|0?[469])([-\/\._])(30|[12][0-9]|0?[1-9])$)|(^((1[8-9]\d{2})|([2-9]\d{3}))([-\/\._])(0?2)([-\/\._])(2[0-8]|1[0-9]|0?[1-9])$)|(^([2468][048]00)([-\/\._])(0?2)([-\/\._])(29)$)|(^([3579][26]00)([-\/\._])(0?2)([-\/\._])(29)$)|(^([1][89][0][48])([-\/\._])(0?2)([-\/\._])(29)$)|(^([2-9][0-9][0][48])([-\/\._])(0?2)([-\/\._])(29)$)|(^([1][89][2468][048])([-\/\._])(0?2)([-\/\._])(29)$)|(^([2-9][0-9][2468][048])([-\/\._])(0?2)([-\/\._])(29)$)|(^([1][89][13579][26])([-\/\._])(0?2)([-\/\._])(29)$)|(^([2-9][0-9][13579][26])([-\/\._])(0?2)([-\/\._])(29)$))|(^\d{4}年\d{1,2}月\d{1,2}日$)$")
        date=re.compile("((\d{4}|\d{2})(\-|\/|\.)\d{1,2}(\-|\/|\.)\d{1,2})|((\d{4}年)?\d{1,2}月\d{1,2}日)")
        times=re.compile('(\d{1,2}:\d{1,2})|((\d{1,2}点\d{1,2}分)|(\d{1,2}时))')
        alpha=re.compile(string.ascii_letters)
        one_line = []
        line=alpha.sub('',line)# 去掉字母
        line=punctuation.sub('',line)#

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

        one_line=set(one_line).difference(stopword)
        if Deal.queue.full():
            time.sleep(0.5)
        # Deal.queue.put([one_line,label])
        return self,label,list(one_line)


if __name__ == '__main__':
    arg=argument()
    logger=log_config()
    deal=Deal(arg,logger)


