# coding:utf-8

import os,re
import xml.sax
import xml.sax.handler
from collections import defaultdict
from xml.sax._exceptions import *
from xml.sax import xmlreader, saxutils, handler


files=['exp_polarity.txt','gate_default','intensifiers','markup.txt','sent_rule.txt','sent_subj.txt','subjclueslen1polar',
       'subjcluesSentenceClassifiersOpinionFinderJune06','valenceshifters']


class WorkersHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.current_data=""
        self.names=''
        self.author=''
        self.sent=[]

    def startElement(self,tag,attributes):
        self.current_data=tag
        if tag=="MPQASENT":
            print(attributes['autoclass1'])
        elif tag=="MPQAPOL":
            print(attributes['autoclass'])

    def endElement(self,tag):
        if self.current_data=='MPQASENT':
            print(self.sent[-1])


    def characters(self,content):
        if self.current_data=='MPQAPOL':
            self.sent[-1]+=content
        elif self.current_data=='MPQASENT':
            self.sent.append(content)

class ErrorHandler:
    def __init__(self, parser):
        self.parser = parser

    def fatalError(self, msg):
        print(msg)
        handler=WorkersHandler()
        self.parser.setContentHandler(handler)
        # add you handling here

import xml.etree.ElementTree as ET

class ParseXml():
    def __init__(self,files):
        self.files=files
        self.read()

    def read(self):
        with open(self.files,'r') as f:
            data=f.read()
        data="<root>"+data+"</root>"
        tree=ET.fromstring(data)
        tree=ET.ElementTree(tree)
        self.root=tree.getroot()

    def iter(self):
        sents=[]
        for child_root in self.root:
            attrib=child_root.attrib
            sentclass=attrib['autoclass2']
            sent=re.sub("[\n\s]+",' '," ".join(child_root.itertext()))
            word_polar=[]
            for child in child_root:
                word_polar.append({child.attrib:child.findtext('.')})
            sents.append({sentclass:sent,"polar":word_polar})
        return sents


class MarkWain():
    def __init__(self):
        self.root='C:\\Users\Administrator\Desktop\情感分析和观点提取\opinion finder\database\docs\marktwain'

    def read(self,files,line=False):
        with open( files,'r') as f:
            if line:
                data=f.readlines()
            else:
                data=f.read()
        return data

    def iten(self):
        filesname={}
        suffix='_auto_anns'
        for i in os.listdir(self.root):
            path=os.path.join(self.root,i)
            if os.path.isfile(path):
                data=self.read(path)
                filesname.update({i+suffix:data})
        return filesname

    def GateDefault(self,data,sourcedt):
        for dt in data:
            lines=dt.split()[1:]
            indexs=lines[0].strip().split(",")
            s,e=int(indexs[0]),int(indexs[1])
            if len(lines)<=2:
                continue
            words=re.findall('string=\"(.+?)\"',dt)[0]

    def sent(self,data,source):
        """
        exp_polarity.txt:极性分类器基于在MPQA语料库上训练的模型，用它们的上下文极性标记文档中的单词。
        sent_rule.txt: 这是基于规则的主观性分类器依赖于手工制作的规则来标记句子在文件中作为主观或客观，具有高精度和低召回率。
        sent_subj.txt：主观性分类器基于在MPQA语料库上训练的模型将文档中的句子标记为主观或客观。
        :param data:
        :param source:
        :return:
        """
        labels=defaultdict(list)
        for dt in data:
            lines=dt.split()
            index=lines[0].split("_")
            s,e=int(index[-2]),int(index[-1])
            labels[lines[-1]].append(source[s:e])
        print(labels.keys())
        return labels

    def sents(self,data,source):
        """
        subjclueslen1polar,subjcluesSentenceClassifiersOpinionFinerJune06,valenceshifters,:在文本文档中可以找到用于识别主观句子和情感表达的线索。
        :param data:
        :param source:
        :return:
        """
        labels=defaultdict(list)
        a=[]
        for dt in data:
            lines=dt.split()[1:]
            index=lines[0].split(",")
            s,e=int(index[-2]),int(index[-1])
            types=lines[2].split("=")[-1][1:-1]
            labels[types].append(source[s:e])
        for k,v in labels.items():
            print(k,v)


    def iter(self):
        filesdict=self.iten()
        for filesname,data in filesdict.items():
            secondpath=os.path.join(self.root,filesname)
            for f in os.listdir(secondpath):
                path=os.path.join(secondpath,f)
                if f=='markup.txt':
                    xml=ParseXml(path)
                    xml.iter()
                elif f=="gate_default":
                    newdta=self.read(path,line=True)
                    print(newdta)



if __name__ == '__main__':
    parser=xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces,0)
    handler=WorkersHandler()
    parser.setContentHandler(handler)
    # errorHandler=ErrorHandler(parser)
    # parser.setErrorHandler(errorHandler)
    # parser.parse(files)

    ab='C:\\Users\Administrator\Desktop\情感分析和观点提取\opinion finder\database\docs\marktwain\\letters41'
    path="C:\\Users\Administrator\Desktop\情感分析和观点提取\opinion finder\database\docs\marktwain\\letters41_auto_anns"
    files=os.path.join(path,'intensifiers')

    mark=MarkWain()
    dt=mark.read(files,line=True)
    sourece=mark.read(ab)
    mark.sents(dt,sourece)


