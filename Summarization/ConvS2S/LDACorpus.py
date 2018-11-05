# coding:utf-8
"""生成主题语料库"""
from sklearn.decomposition import LatentDirichletAllocation,NMF
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.externals import joblib
from database import *
import jieba,re,os
from multiprocessing import Pool,cpu_count


class LDAModel():
    def __init__(self,content,n_features=10000,n_topics=10,n_top_words=200,lda=True,tfidf=False):
        """
        :param n_features: 特征关键字
        :param n_topics: 主题数
        :param n_top_words:
        :param lda:是否使用lda模型或者NMF模型
        :param tfidf: 是否使用tfidf特征或者tf特征
        :return:
        """
        self.n_top_words=n_top_words
        if tfidf:
            tfidf_vectorizer = TfidfVectorizer(strip_accents='unicode',
                                               max_features=n_features,
                                               max_df=0.5,
                                               min_df=10)
            vectorizer=tfidf_vectorizer
        else:
            tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                            max_features=n_features,
                                            max_df=0.5,
                                            min_df=10)
            vectorizer=tf_vectorizer
        # 特征向量
        self.feature_vectorizer=vectorizer.fit_transform(content)
        # 特征向量名
        self.feature_names=vectorizer.get_feature_names()

        if lda:
            lda = LatentDirichletAllocation(n_components=n_topics,
                                            max_iter=1000,  # 迭代次数
                                            learning_method='online',
                                            learning_offset=50.,
                                            random_state=0)
            model=lda
        else:
            nhm = NMF(n_components=n_topics, random_state=1, alpha=1, l1_ratio=0.5, max_iter=1000)
            model=nhm

        # 训练模型
        model.fit(self.feature_vectorizer)
        self.model=model
        joblib.dump(self.model,"./model/LDA.m")

    def get_topic_distrbution(self):
        return self.model.components_

    def get_topic_words(self):
        corpus=[]
        for topic_idx,topic in enumerate(self.model.components_):
            # 每个主题前n_top_words个主题词
            topic_words=",".join([self.feature_names[i] for i in topic.argsort()[:-self.n_top_words-1:-1]])
            corpus.append(topic_words)
        return corpus

    def get_doc_topic(self):
        """获得每篇文章的主题分布情况"""
        doc_topic=self.model.transform(self.feature_vectorizer)
        return doc_topic

    def get_model(self):
        return self.model

def stopword():
    with open('../data/stopwords.txt','r') as f:
        words=f.readlines()
        words=[word.strip() for word in words]
    return words

def cut_words(content):
    re_compile=re.compile("[A-Za-z0-9\s]")
    content=jieba.cut(content)
    content=[word for word in content if not re_compile.match(word)]
    stopwords=stopword()
    content=[word for word in content if word not in stopwords]
    return content

def main():
    contentwords = []
    if not os.path.isfile('./corpus.txt'):
        mongo=MySQL()
        mongo.login()
        cursor=mongo.get_cursor()
        sent="select content from news where 1"
        cursor.execute(sent)
        pool=Pool(cpu_count()-1)
        for content in cursor.fetchall():
            words = pool.apply_async(func=cut_words, args=(content[0],))
            contentwords.append(words.get())

        print(len(contentwords))
        contentwords=[" ".join(word) for word in contentwords]
        with open("./corpus.txt",'w') as f:
            for content in contentwords:
                f.write(content+"\n")
    else:
        with open("./corpus.txt",'r') as f:
            contents=f.readlines()
            for content in contents:
                contentwords.append(content)
    # look_best_topic_num(contentwords)
    lda=LDAModel(contentwords)
    # distrbution=lda.get_topic_distrbution()
    # print(distrbution)
    doc_topic=lda.get_topic_words()
    corpus=set()
    for topic in doc_topic:
        for a in topic.split(','):
            corpus.add(a)
    f=open('../data/topic_vocab.txt','w')
    for cor in corpus:
        f.write(cor+"\n")


def look_best_topic_num(content):
    n_topics=range(10,40,3)
    perplexityLst = [1.0] * len(n_topics)
    lda_models = []
    tfidf_vectorizer= TfidfVectorizer(strip_accents='unicode',
                                       max_features=20000,
                                       max_df=0.5,
                                       min_df=10)
    tfidf = tfidf_vectorizer.fit_transform(content)
    for idx, n_topic in enumerate(n_topics):
        lda = LatentDirichletAllocation(n_components=n_topic,
                                        max_iter=1000,
                                        learning_method='batch',
                                        evaluate_every=200,
                                        verbose=0)
        lda.fit(tfidf)
        # 计算模型的困惑度
        perplexityLst[idx] = lda.perplexity(tfidf)
        lda_models.append(lda)
    best_index = perplexityLst.index(min(perplexityLst))
    best_n_topic = n_topics[best_index]
    print("Best # of Topic: ", best_n_topic)


if __name__ == '__main__':
    main()





