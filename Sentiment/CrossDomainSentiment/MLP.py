# coding:utf-8
"""
多层感知机
"""
import numpy as np
import theano.tensor as T
import theano,sys
import os,gzip,time
import pickle as cPickle

class Perceptron():
    """
    感知机
    """
    def __init__(self,learning_rate,max_iter):
        self.lr=learning_rate
        self.iter=max_iter

    def train(self,feature,labels):
        self.w=np.zeros(feature.shape[1]+1)
        correct_count=0

        num=0
        while num<self.iter:
            # 随机寻找样本
            index=np.random.randint(0,len(labels)-1)
            x=feature[index,:]
            x=np.insert(x,feature.shape[1],1.0)
            y=2*labels[index]-1
            wx=np.dot(self.w,x)

            if wx*y>0:
                correct_count+=1
                if correct_count>self.iter:
                    break
                continue

            self.w+=self.lr*y*x
    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        return int(wx > 0)

    def predict(self,feature):
        feature=np.insert(feature,feature.shape[1],1.0,axis=1)
        pre=np.dot(self.w,feature.T)
        lables=(pre>0)+0
        return lables


class HiddenLayer():
    def __init__(self,rng,input,n_in,n_out,W=None,b=None,activate=T.tanh):
        self.input=input
        # 代码要兼容GPU，则必须使用 dtype=theano.config.floatX,并且定义为theano.shared
        if W is None:
            w_value=np.asarray(rng.uniform(low=-np.sqrt(6./(n_in+n_out)),
                                           high=np.sqrt(6./(n_in+n_out)),
                                           size=(n_in,n_out)),dtype=theano.config.floatX)
            if activate==T.nnet.sigmoid:
                w_value*=4
            # 将变量设置为全局变量，其值可以在多个函数中共用．
            # borrow=True/False:对数据的任何改变会/不会影响到原始的变量
            W=theano.shared(value=w_value,name='W',borrow=True)

        if b is None:
            b_value=np.zeros((n_out,),dtype=theano.config.floatX)
            b=theano.shared(b_value,name='b',borrow=True)

        self.W=W
        self.b=b
        lin_out=T.dot(input,self.W)+self.b
        self.output=(lin_out if activate is None else activate(lin_out))
        self.params=[self.W,self.b]

class LogisticRegression():
    def __init__(self,input,n_in,n_out):
        """
        :param input: shape is (batch,n_in)
        :param n_in: 上一层(隐含层)的输出
        :param n_out: 输出的类别数
        """
        self.W=theano.shared(value=np.zeros((n_in,n_out),dtype=theano.config.floatX),name='W',borrow=True)
        self.b=theano.shared(value=np.zeros((n_out,),dtype=theano.config.floatX),name='b',borrow=True)
        self.p_y_given_x=T.nnet.softmax(T.dot(input,self.W)+self.b)# 点乘得到的维度为(batch,n_out)
        self.y_pred=T.argmax(self.p_y_given_x,axis=1)
        self.params=[self.W,self.b]

    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

    def error(self,y):
        if y.ndim!=self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype.startwith("int"):
            return T.mean(T.neg(self.y_pred,y))
        else:
            raise NotImplementedError

class MLP():
    """
    多层感知机
    """
    def __init__(self,rng,input,n_in,n_hidden,n_out):
        """
        :param rng: 随机数种子
        :param input: 输入的数据
        :param n_in: 单个样本数据的长度
        :param n_hidden: 规定的隐藏层大小
        :param n_out: 输出的类别
        """
        self.hiddenlayer=HiddenLayer(rng,input,n_in,n_out)
        # 将隐含层hiddenLayer的输出作为分类层logRegressionLayer的输入
        self.logRegressionLayer=LogisticRegression(self.hiddenlayer.output,n_hidden,n_out)
        # L1正则化
        self.L1=(abs(self.hiddenlayer.W).sum()+abs(self.logRegressionLayer.W).sum())
        self.L2_sqr=((self.hiddenlayer.W**2).sum()+(self.logRegressionLayer.W**2).sum())

        self.negative_log_likelihood=(self.logRegressionLayer.negative_log_likelihood)
        self.errors=(self.logRegressionLayer.error)

        self.params=self.hiddenlayer.params+self.logRegressionLayer.params

def load_data(dataset):
    data_dir,data_file=os.path.split(dataset)
    if data_dir=='' and not os.path.isfile(dataset):
        new_path=os.path.join(os.path.split(__file__)[0],"..","data",dataset)
        if os.path.isfile(new_path) or data_file=="mnist.pkl.gz":
            dataset=new_path
            data_dir,data_file=os.path.split(dataset)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

    if (not os.path.isfile(dataset)) and data_file=='mnist.pkl.gz':
        from urllib.request import urlretrieve
        origin= 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from %s' % origin)
        urlretrieve(origin,dataset)
    print('... loading data')

    f=gzip.open(dataset,'rb')
    train_set,valid_set,test_set=cPickle.load(f,encoding='bytes')
    f.close()
    def shared_dataset(data_xy,borrow=True):
        """
        将数据设置成shared variables，主要时为了GPU加速，只有shared variables才能存到GPU memory中
        :param data_xy:
        :param borrow:
        :return:
        """
        data_x,data_y=data_xy
        shared_x=theano.shared(value=np.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
        shared_y=theano.shared(value=np.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
        return shared_x,T.cast(shared_y,"int32")

    train_set_x,train_set_y=shared_dataset(train_set)
    test_set_x,test_set_y=shared_dataset(test_set)
    valid_set_x,valid_set_y=shared_dataset(valid_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def test_mlp(learn_rating=0.01,L1_reg=0.00,L2_reg=0.0001,n_epochs=1000,dataset="mnist.pkl.gz",batch_size=20,n_hidden=200):
    datasets=load_data(dataset)
    train_set_x,train_set_y=datasets[0]
    valid_set_x,valid_set_y=datasets[1]
    test_set_x,test_set_y=datasets[2]

    n_train_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_valid_batches=valid_set_x.get_value(borrow=True).shape[0]/batch_size
    n_test_batches=test_set_x.get_value(borrow=True).shape[0]/batch_size

    # index表示batch的下标，标量
    index=T.iscalar()
    # x表示数据集
    x=T.matrix("x")
    # y表示类别，一维向量
    y=T.ivector("y")
    rng=np.random.RandomState(12345)

    classifier=MLP(rng,input=x,n_in=28*28,n_hidden=n_hidden,n_out=10)

    # 代价函数，有规则化项 ，用y来初始化，而其实还有一个隐含的参数x在classifier中
    cost=(classifier.negative_log_likelihood(y)+L1_reg*classifier.L1+L2_reg*classifier.L2_sqr)

    # 在function被调用时，x、y将被具体地替换为它们的value，而value里的参数index就是inputs=[index]这里给出。
    # 比如test_model(1)，首先根据index=1具体化x为test_set_x[1 * batch_size: (1 + 1) * batch_size]，
    # 具体化y为test_set_y[1 * batch_size: (1 + 1) * batch_size]。然后函数计算outputs=classifier.errors(y)，
    # 这里面有参数y和隐含的x，所以就将givens里面具体化的x、y传递进去。
    test_model=theano.function(inputs=[index], outputs=classifier.errors(y), givens={x:test_set_x[index*batch_size:(index+1)*batch_size],
                                                                                  y:test_set_y[index*batch_size:(index+1):batch_size]})

    validate_model=theano.function(inputs=[index],outputs=classifier.errors(y),givens={x:valid_set_x[index*batch_size:(index+1)*batch_size],
                                                                                       y:valid_set_y[index*batch_size:(index+1)*batch_size]})

    # cost函数对各个参数的偏导数值，即梯度，存于gparams
    gparams=[T.grad(cost,param) for param in classifier.params]

    #参数更新规则
    updates=[(param,param-learn_rating*gparam) for param,gparam in zip(classifier.params,gparams)]

    train_model=theano.function(inputs=[index],outputs=cost,updates=updates,givens={ x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                                                                     y:train_set_y[index*batch_size: (index+1)*batch_size]})

    patience=10000
    patience_increase=2
    # 提高的阈值，在验证误差减小到之前的0.995倍时，会更新best_validation_loss
    improvement_threshold=0.995
    # 这样设置validation_frequency可以保证每一次epoch都会在验证集上测试。
    validation_frequency = min(n_train_batches, patience / 2)
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch=0
    done_looping=False

    while epoch<n_epochs and (not done_looping):
        epoch+=1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost=train_model(minibatch_index)
            iters=(epoch-1)*n_train_batches+minibatch_index
            if (iters+1)%validation_frequency==0:
                validation_losses=[validate_model[i] for i in range(n_valid_batches)]
                validation_loss=np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f' % (epoch,minibatch_index + 1,n_train_batches,validation_loss * 100.))

                if validation_loss<best_validation_loss:
                    if validation_loss<best_validation_loss*improvement_threshold:
                        patience=max(patience,iters*patience_increase)

                        best_validation_loss=validation_loss
                        best_iter=iters
                    test_losses=[test_model(i) for i in range(n_test_batches)]
                    test_score=np.mean(test_losses)
                    print(('epoch %i, minibatch %i/%i, test error of best model %f %%')%(epoch,minibatch_index+1,n_train_batches,test_score*100.))
            if patience<=iters:
                done_looping=True
                break

    end_time=time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    load_data("mnist.pkl.gz")














