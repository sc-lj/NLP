# coding:utf-8

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.externals import joblib
from Data import *

def lda(content):
    n_features=10000 #特征关键字数
    n_topics = 10 #主题数
    n_top_words=200
    tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                    max_features=n_features,
                                    stop_words='english',
                                    max_df=0.5,
                                    min_df=10)
    tf = tf_vectorizer.fit_transform(content)
    tf_feature_names=tf_vectorizer.get_feature_names()

    lda = LatentDirichletAllocation(n_topics=n_topics,
                                    max_iter=1000,#迭代次数
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    joblib.dump(lda,'../model/lda.m')
    for topic_idx,topic in enumerate(lda.components_):
        topic_words=",".join([tf_feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]])


