# coding:utf-8

"""
Central Moment Discrepancy(中心力矩差异) 算法的实现
"""

import theano.tensor as T
from keras import backend as K
from keras import regularizers

class CMDRegularizer(regularizers):
    def __init__(self,l=1.,n_moments=5,name='mmd',beta=1):
        self.use_learning_phase=1
        self.l=l
        self.n_moments=n_moments
        self.name=name
        self.beta=beta

    def set_layer(self,layer):
        self.layer=layer

    def __call__(self, loss):
        if not hasattr(self,"layer"):
            raise Exception("在调用该实例前，需要在ActivityRegularizer 实例中调用’set_layer'")
        regularizer_loss=loss

        if len(self.layer.inbound_nodes)>1:
            if self.name=='mmd':
                sim=self.mmd(self.layer.get_output_at(0),
                        self.layer.get_output_at(1),
                        self.beta)
            elif self.name=='mmatch':
                sim=self.mmatch(self.layer.get_output_at(0),
                        self.layer.get_output_at(1),
                        self.n_moments)
            elif self.name=='mmd5':
                sim=self.mmdK(self.layer.get_output_at(0),
                              self.layer.get_output_at(1),
                              5)
            elif self.name=='mmatchK':
                sim=self.mmatch(self.layer.get_output_at(0),
                                self.layer.get_output_at(1),
                                self.beta)
            else:
                raise ValueError("请输入mmd、mmatch、mmd5、mmatchK，其中之一")

            add_loss=T.switch(T.eq(len(self.layer.inbound_nodes),2),sim,0)
            regularizer_loss+=self.l*add_loss
            return K.in_train_phase(regularizer_loss,loss)

    def get_config(self):
        return {'name':self.__class__.__name__,'l':float(self.l)}

    @classmethod
    def from_config(cls,config):
        return cls(**config)

    def mmd(self,x1,x2,beta):
        """
        实现高斯核距离函数，是一种非线性的欧几里得距离计算函数
        :param x1:
        :param x2:
        :param beta:
        :return:
        """
        x1x1=self.gaussian_kernel(x1,x1,beta)
        x1x2=self.gaussian_kernel(x1,x2,beta)
        x2x2=self.gaussian_kernel(x2,x2,beta)
        diff=x1x1.mean()-2*x1x2.mean()+x2x2.mean()
        return diff

    def gaussian_kernel(self,x1,x2,beta=0.5):
        r=x1.dimshuffle(0,'x',1) # 转换数据维度，r的维度是[x1[0],1,x1[2]]
        return T.exp(-beta*T.sqrt(r-x2).sum(axis=-1))

    def mmdK(self,x1,x2,moment):
        m1=x1.mean(0)
        m2=x2.mean(0)
        s1=m1
        s2=m2
        for i in range(moment-1):
            s1+=(x1**T.cast(i+2,"int32")).mean(0)
            s2+=(x2**T.cast(i+2,'int32')).mean(0)
        return self.matchnorm(s1,s2)

    def mmatch(self,x1,x2,moments=5):
        mx1=x1.mean(0)
        mx2=x2.mean(0)
        sx1=x1-mx1
        sx2=x2-mx2
        dm=self.matchnorm(mx1,mx2)
        scms=dm
        for i in range(moments-1):
            # 中心样本的moment差分
            scms+=self.moment_diff(sx1,sx2,i+2)
        return scms

    def matchnorm(self,x1,x2):
        """
        标准的欧几里德规范
        """
        return ((x1-x2)**2).sum().sqrt()

    def moment_diff(self,sx1,sx2,k):
        """两个元素的差"""
        ss1=(sx1**T.cast(k,"int32")).mean(0)
        ss2=(sx2**T.cast(k,"int32")).mean(0)
        return self.matchnorm(ss1,ss2)

