# coding:utf-8

import os,re,json,jieba
from collections import defaultdict
doubelsurname=['欧阳', '太史', '端木', '上官', '司马', '东方', '独孤', '南宫', '万俟', '闻人', '夏侯', '诸葛', '尉迟', '公羊', '赫连', '澹台', '皇甫', '宗政', '濮阳', '公冶', '太叔', '申屠', '公孙', '慕容', '仲孙', '钟离', '长孙', '宇文', '司徒', '鲜于', '司空', '闾丘', '子车', '亓官', '司寇', '巫马', '公西', '颛孙', '壤驷', '公良', '漆雕', '乐正', '宰父', '谷梁', '拓跋', '夹谷', '轩辕', '令狐', '段干', '百里', '呼延', '东郭', '南门', '羊舌', '微生', '公户', '公玉', '公仪', '梁丘', '公仲', '公上', '公门', '公山', '公坚', '左丘', '公伯', '西门', '公祖', '第五', '公乘', '贯丘', '公皙', '南荣', '东里', '东宫', '仲长', '子书', '子桑', '即墨', '达奚', '褚师', '吴铭']

class Corpus():
    def __init__(self,root_path):
        self.path=root_path
        self.corpus='../data/BosonNLP_NER.txt'
        self.ner='../data/NER.json'
        self.files = './data/BosonNLP_NER_6C.txt'
        self.entity_tag = {"TIME": "time", "LOC": "location", "PER": "person_name", "ORG": "org_name", "COMP": "company_name","PRO": "product_name"}

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

    def drop_tag(self,seqence, tag):
        compiles = re.compile("{{(.*?)}}")
        filter_compile = re.compile("[a-zA-Z_]+:")
        seqences = compiles.split(seqence)
        seqs = []
        for seq in seqences:
            if len(seq.strip()) == 0:
                continue
            if filter_compile.match(seq):
                name, entity = seq.split(":", 1)
                if name == self.entity_tag[tag]:
                    seqs.append(" START " + entity + " END ")
                else:
                    seqs.append(entity)
            else:
                seqs.append(seq)
        return "".join(seqs)

    def cut_seq(self,seqence):
        compile_url = re.compile("(http[s]?:[a-zA-Z0-9\.\/\?#&\+=]+)")
        seqence = compile_url.sub("URL", seqence)
        cuts = list(jieba.cut(seqence, HMM=False))
        cut = [a.strip() for a in cuts if len(a.strip()) != 0]
        return cut

    def find_tag(seqs, tag="START"):
        if isinstance(tag, list):
            index = [ind for ind, word in enumerate(seqs) if word in tag]
        else:
            index = [ind for ind, word in enumerate(seqs) if word == tag]
        return index

    def perEntity(self,start, end, seqs, tag):
        xrange = end - start - 1
        piece_tag = seqs[start + 1:end]
        if xrange == 1:
            tag.append('J-PER')
        elif xrange == 2:
            if len(piece_tag[0]) >= 2:
                if piece_tag[0] in doubelsurname:
                    tag.append("C-PER")
                else:
                    tag.append("I-PER")

                if len(piece_tag[1]) == 2:
                    tag.append("K-PER")
                else:
                    tag.append("E-PER")
            else:
                tag.append("C-PER")
                if len(piece_tag[1]) >= 2:
                    tag.append("K-PER")
                else:
                    tag.append("F-PER")
        else:
            if len(piece_tag[0]) >= 2:
                if piece_tag[0] in doubelsurname:
                    tag.append("C-PER")
                else:
                    tag.append("I-PER")
            else:
                tag.append("C-PER")

            for i in range(1, len(piece_tag) - 1):
                if len(piece_tag[i]) >= 2:
                    tag.append("K-PER")
                else:
                    tag.append("D-PER")
            if len(piece_tag[-1]) >= 2:
                tag.append("K-PER")
            else:
                tag.append("E-PER")

    def genEntityCorpus(self,entype):
        assert entype in self.entity_tag
        lines = open(self.files, 'r').readlines()
        tags=[]
        words=[]
        for line in lines:
            seqences = self.drop_tag(line, entype)
            seqs = self.cut_seq(seqences)
            init_seq = re.sub("START|END", '', seqences)
            init_seqs = self.cut_seq(init_seq)

            tag = []
            index = self.find_tag(seqs, tag=["START", "END"])
            index = sorted(index)
            reduce = []
            for i in range(0, len(index), 2):
                reduce.append(index[i + 1] - index[i])
            j = -1
            i = 0
            while j < len(seqs) - 1:
                j += 1
                if seqs[j] == "START":
                    if len(tag) > 0:
                        if tag[-1] != "X-PER" and tag[-1] == "Z-PER":
                            tag[-1] = "A-PER"
                    start = j
                    j = end = reduce[i] + start
                    self.perEntity(start, end, seqs, tag)
                    i += 1
                    if len(reduce) > i:
                        if seqs[j + 1] == "START":
                            continue
                        if seqs[j + 2] == 'START':
                            tag.append("X-PER")
                        else:
                            tag.append('B-PER')
                        j += 1
                    else:
                        if len(seqs) > j + 1:
                            tag.append('B-PER')
                            j += 1
                else:
                    tag.append("Z-PER")
            tags.append(tag)
            words.append(init_seqs)
        return tags,words


if __name__ == '__main__':
    corpus=Corpus('../data/BosonNLP_NER_6C.txt')
    # corpus.gen_ner_corpus()
    name = ["A", "B", "C", "D", "E", "F", "I", "J", "K", "X", "Z"]



