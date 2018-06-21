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

    def read_corpus(self):
        with open(self.gloconf.target_file,'r',encoding='utf-8') as f:
            data=f.readline()
            while data:
                newdata=json.loads(data)
                label,content=newdata['label'],newdata['content']
                yield label,content
                data=f.readline()


