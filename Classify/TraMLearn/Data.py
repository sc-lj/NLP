# coding:utf-8

from KwordExtr.SKE.main import SemankeyWord
from KwordExtr.TextRank.TextRankKeyword import TextRankKeyword
from multiprocessing import Pool,cpu_count

def TRKeyWord(line,trkeyword):
    lable, content = line.split(maxsplit=1)
    keyword =trkeyword.get_keywords(content, 10, 1,)
    words = " ".join([item['word'] for item in keyword])
    return [lable,words]

def genKeyWords(files):
    with open(files,'r') as f:
        lines=f.readlines()
        data=[]
        trkeyword = TextRankKeyword(stop_words_file="../data/stopwords.txt")
        pool=Pool(10)
        for line in lines:
            keyword=pool.apply_async(func=TRKeyWord,args=(line,trkeyword))
            words=keyword.get()
            data.append(words)

    newdata=[a[1] for a in data]
    labels=[a[0] for a in data]
    return newdata,labels
