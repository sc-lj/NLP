# coding:utf-8

import os,sys
path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(path))
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.grid_search import GridSearchCV
import numpy as np
from Data import genKeyWords

filename=['../data/cnews.test.txt','../data/cnews.train.txt','../data/cnews.val.txt','../data/cnews.vocab.txt']

def svc():
    count=CountVectorizer(max_df=0.9,max_features=10000)
    tfidf=TfidfTransformer()
    _svc=SVC(C=0.99,kernel="linear")
    train_data, train_label=genKeyWords("../data/cnews.train.txt")
    test_data,test_label=genKeyWords("../data/cnews.test.txt")

    pipline=Pipeline([("count",count),("tfidf",tfidf),("svc",_svc)])

    pipline=pipline.fit(train_data,train_label)
    predicted = pipline.predict(test_data)
    print('SVC', np.mean(predicted == test_label))

def searchParam():
    """调参数"""
    count = CountVectorizer(max_df=0.9, max_features=10000)
    tfidf = TfidfTransformer()
    _svc = SVC(C=0.99, kernel="linear")
    train_label, train_data = genKeyWords("../data/cnews.train.txt")
    test_label, test_data = genKeyWords("../data/cnews.test.txt")
    label=train_label+test_label
    data=train_data+test_data

    parameters = {'count__max_df': (0.4, 0.5, 0.6, 0.7), 'count__max_features': (None, 5000, 10000, 15000),
                  'tfidf__use_idf': (True, False)}
    pipline=Pipeline([("count",count),("tfidf",tfidf),("svc",_svc)])
    grid_search = GridSearchCV(pipline, parameters, n_jobs=-1, verbose=1)
    grid_search.fit(data,label)

    best_parameters=grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))



if __name__ == '__main__':
    # svc()
    searchParam()