# coding:utf-8

from sklearn.feature_extraction import DictVectorizer

from KwordExtr.SKE.main import SemankeyWord

def corpus(files):
    with open(files,'r') as f:
        lines=f.readlines()
        lables=[]
        keywords=[]
        for line in lines:
            try:
                lable,title,content=line.split(maxsplit=2)
                lables.append(lable)
                keyword=SemankeyWord(content=content,title=title)
                keywords.extend(keyword.split(","))
            except:
                print(line)
                continue
    return lables,keywords

filename=['../data/cnews.test.txt','../data/cnews.train.txt','../data/cnews.val.txt','../data/cnews.vocab.txt']

def gencorpus():
    keywords=[]
    for files in filename:
        lable,keyword=corpus(files)
        keywords.extend(keyword)

    with open('./corpus.txt','w') as f:
        for keyword in keywords:
            f.write(keyword)





if __name__ == '__main__':
    gencorpus()

