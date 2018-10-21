# coding:utf-8

import os,re,json,jieba
from collections import defaultdict
doubelsurname=['欧阳', '太史', '端木', '上官', '司马', '东方', '独孤', '南宫', '万俟', '闻人', '夏侯', '诸葛', '尉迟', '公羊', '赫连', '澹台', '皇甫', '宗政', '濮阳', '公冶', '太叔', '申屠', '公孙', '慕容', '仲孙', '钟离', '长孙', '宇文', '司徒', '鲜于', '司空', '闾丘', '子车', '亓官', '司寇', '巫马', '公西', '颛孙', '壤驷', '公良', '漆雕', '乐正', '宰父', '谷梁', '拓跋', '夹谷', '轩辕', '令狐', '段干', '百里', '呼延', '东郭', '南门', '羊舌', '微生', '公户', '公玉', '公仪', '梁丘', '公仲', '公上', '公门', '公山', '公坚', '左丘', '公伯', '西门', '公祖', '第五', '公乘', '贯丘', '公皙', '南荣', '东里', '东宫', '仲长', '子书', '子桑', '即墨', '达奚', '褚师', '吴铭']

class Corpus():
    def __init__(self,root_path):
        self.path=root_path
        self.corpus='../data/BosonNLP_NER.txt'
        self.ner='../data/NER.json'

    def read_corpus(self):
        """去掉命名实体标志"""
        files=open(self.path,'r')
        lines=files.readlines()
        compiles_1 = re.compile("{{.*?:")
        compiles_2 = re.compile("}}")
        newlines=[]
        for line in lines:
            newdata=compiles_1.sub("",line)
            newdata=compiles_2.sub("",newdata)
            if len(newlines)!=0 and newdata==newlines[-1]:
                continue
            newlines.append(newdata)
        f=open(self.corpus,'w')
        f.writelines(newlines)
        f.close()

    def gen_ner_corpus(self):
        """产生各类命名实体集"""
        f=open(self.path,'r')
        data=f.read()
        compile_3=re.compile("{{(.*?)}}")
        newdata=compile_3.findall(data)
        nerdict=defaultdict(list)

        for da in newdata:
            name,entity=da.split(":",maxsplit=1)
            if name =="location":
                nerdict["loc"].append(entity)
            elif name=="person_name":
                nerdict["person"].append(entity)
            elif name in ['company_name','org_name']:
                nerdict['org'].append(entity)
            else:
                pass

        nerdict=dict(nerdict)
        for key,value in nerdict.items():
            nerdict[key]=list(set(value))
        nerdict=json.dumps(nerdict)
        f=open(self.ner,'w')
        f.write(nerdict)
        f.close()

    def _static(self):
        f=open(self.path,'r')
        lines=f.readlines()
        compil=re.compile("}}|{{")
        newlines=[]
        for line in lines:
            wordline=[]
            words=compil.split(line)
            for word in words:
                word=word.strip()
                if len(word)==0:
                    continue
                wordlist=word.split(":",maxsplit=1)
                if len(wordlist)==2:
                    name, entity=wordlist

    def _judge(self,words,role):
        name, entity = words
        if name==role:
            for i,word in enumerate(words):
                pass



    def static_(self,char_mark,mark):
        """根据提供的角色标志和角色，给文本做上相应的标记"""
        assert mark in ['loc','person','org']
        ner=open(self.ner,'r')
        nerdata=ner.read()
        nerdata=json.loads(nerdata)
        markdt=nerdata[mark]

        f=open(self.corpus,'r')
        lines=f.readlines()
        newlines=[]
        compile_4=re.compile("\s|' '")
        for line in lines:
            line=compile_4.sub('',line)
            cuts=list(jieba.cut(line))
            for cut in cuts:

                pass




if __name__ == '__main__':
    corpus=Corpus('../data/BosonNLP_NER_6C.txt')
    # corpus.gen_ner_corpus()
    name = ["A", "B", "C", "D", "E", "F", "I", "J", "K", "X", "Z"]
    corpus._static()

