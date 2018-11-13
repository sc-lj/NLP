# coding:utf-8

from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import auc,roc_curve
import numpy as np
from Data import genKeyWords


def svc():
    count=CountVectorizer(max_df=0.9,max_features=10000)
    tfidf=TfidfTransformer()
    _svc=SVC(C=0.99)
    train_data, train_label=genKeyWords("../data/cnews.train.txt")
    test_data,test_label=genKeyWords("../data/cnews.test.txt")
    pipline=Pipeline([("count",count),("tfidf",tfidf),("svc",_svc)])

    # pipline=pipline.fit(train_data,train_label)
    # predicted = pipline.predict(test_data)
    # print('SVC', np.mean(predicted == test_label))
    label = train_label + test_label
    data = train_data + test_data
    parameters = {'count__max_df': (0.4, 0.5, 0.6, 0.7), 'count__max_features': (None, 5000, 10000, 15000),
                  'tfidf__use_idf': (True, False),"svc__kernel":('rbf',"linear","sigmoid")}
    grid_search = GridSearchCV(pipline, parameters, n_jobs=-1, verbose=1)
    grid_search.fit(data, label)

    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def Knn():
    train_data, train_label=genKeyWords("../data/cnews.train.txt")
    vectors=CountVectorizer()
    tfidf=TfidfTransformer()
    clf=KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
    pipline=Pipeline([("vectors",vectors),("tfidf",tfidf),("clf",clf)])
    # parameters={'clf__n_neighbors':list(range(5,20,2))}
    # grid_search = GridSearchCV(pipline, parameters, n_jobs=-1, verbose=1)
    # grid_search.fit(train_data,train_label)
    # best_parameters = grid_search.best_estimator_.get_params()
    # for param_name in sorted(parameters.keys()):
    #     print("\t%s: %r" % (param_name, best_parameters[param_name]))

    pipline.fit(train_data,train_label)
    test_data,test_label=genKeyWords("../data/cnews.test.txt")
    predicted = pipline.predict(test_data)
    joblib.dump(pipline,"./knn.m")
    print('KNeighborsClassifier', np.mean(predicted == test_label))


def Bayes(mode='mul'):
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
    joblib.dump(pipline, "./%s_bayes.m"%mode)
    print('naive_bayes', np.mean(predicted == test_label))

def K_Means(minibatch):
    train_data, train_label = genKeyWords("../data/cnews.train.txt")
    vectors = CountVectorizer()
    tfidf = TfidfTransformer()
    if minibatch:
        km = MiniBatchKMeans(n_clusters=10, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=False)
    else:
        km = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=1,
            verbose=False)

    pipline=Pipeline([("vectors",vectors),("tfidf",tfidf),("kmeans",km)])
    pipline.fit(train_data)

def DTrees():
    train_data, train_label = genKeyWords("../data/cnews.train.txt")
    vectors = CountVectorizer()
    tfidf = TfidfTransformer()
    tree=DecisionTreeClassifier(criterion="entropy")
    pipline=Pipeline([("vectors",vectors),("tfidf",tfidf),("tree",tree)])
    params={'vectors__max_df': (0.4, 0.5, 0.6, 0.7), 'vectors__max_features': (None, 500, 1000, 1500),
            'tfidf__use_idf': (True, False),"tree__max_depth":(10,15,20)}
    gridsearch=GridSearchCV(pipline,params,n_jobs=10)
    gridsearch.fit(train_data,train_label)
    best_parameters = gridsearch.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

def Xgboost():
    train_data, train_label = genKeyWords("../data/cnews.train.txt")
    vectors = CountVectorizer()
    tfidf = TfidfTransformer()
    bst=XGBClassifier(n_jobs=10)
    pipline=Pipeline([("vectors",vectors),("tfidf",tfidf),("bst",bst)])
    params={'vectors__max_df': (0.4, 0.5, 0.6, 0.7), 'vectors__max_features': (None, 500, 1000, 1500),
            'tfidf__use_idf': (True, False),"bst__max_depth":list(range(5,20,2)),}
    gride_search=GridSearchCV(pipline,params,n_jobs=10)
    gride_search.fit(train_data,train_label)
    best_parameters =gride_search.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


if __name__ == '__main__':
    # Knn()
    # Bayes()
    # DTrees()
    Xgboost()
