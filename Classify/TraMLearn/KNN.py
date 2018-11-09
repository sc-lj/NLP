# coding:utf-8

from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import numpy as np
from Data import genKeyWords


def Knn():
    train_data, train_label=genKeyWords("../data/cnews.train.txt")
    vectors=CountVectorizer()
    tfidf=TfidfTransformer()
    clf=KNeighborsClassifier(n_neighbors=10,n_jobs=-1)
    pipline=Pipeline([("vectors",vectors),("tfidf",tfidf),("clf",clf)])
    pipline.fit(train_data,train_label)
    test_data,test_label=genKeyWords("../data/cnews.test.txt")
    predicted = pipline.predict(test_data)
    joblib.dump(pipline,"./knn.m")
    print(predicted)
    print(test_label)
    print('KNeighborsClassifier', np.mean(predicted == test_label))


def bayes(mode):
    if mode=='mul':
        model=MultinomialNB()
    elif mode=='gau':
        model=GaussianNB()
    elif mode=='bern':
        model=BernoulliNB()
    else:
        raise ValueError('没有该模式，请填写以下mode，\n mul==>MultinomialNB \ngau==>GaussianNB \n bern==>BernoulliNB')
    train_data, train_label = genKeyWords("../data/cnews.train.txt")
    vectors = CountVectorizer()
    tfidf=TfidfTransformer()
    pipline=Pipeline([("vectors",vectors),("tfidf",tfidf),("bayes",model)])
    pipline.fit(train_data,train_label)
    test_data, test_label = genKeyWords("../data/cnews.test.txt")
    predicted = pipline.predict(test_data)
    joblib.dump(pipline, "./bayes.m")
    print(predicted)
    print(test_label)
    print('naive_bayes', np.mean(predicted == test_label))



if __name__ == '__main__':
    Knn()