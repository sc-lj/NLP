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
    def __init__(self):
        self.gloconf= GloConfi.argument()
        self.conf= Config.argument()

    def read_corpus(self,filename,batch_size=1):
        with open(filename,'r',encoding='utf-8') as f:
            data=f.readlines(batch_size)
            while data:
                labels=[]
                content=[]
                for da in data:
                    jsdata=json.loads(da)
                    label,one_line=jsdata['label'],jsdata['content']
                    labels.append(label)
                    content.append(one_line)
                yield labels,content
                data = f.readlines(batch_size)





