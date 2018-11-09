# coding:utf-8

from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import numpy as np
from Data import genKeyWords


def Knn():
    train_data, train_label=genKeyWords("../data/cnews.train.txt")
    vectors=TfidfVectorizer()
    clf=KNeighborsClassifier(n_neighbors=10,n_jobs=-1)
    pipline=Pipeline([("vectors",vectors),("clf",clf)])
    pipline.fit(train_data,train_label)
    test_data,test_label=genKeyWords("../data/cnews.test.txt")
    predicted = pipline.predict(test_data)
    joblib.dump(pipline,"./knn.m")
    print(predicted)
    print(test_label)
    print('KNeighborsClassifier', np.mean(predicted == test_label))


if __name__ == '__main__':
    Knn()