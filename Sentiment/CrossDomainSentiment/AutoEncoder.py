# coding:utf-8

import numpy as np

class nn():
    def __init__(self,nodes,learn_rate=1.0):
        self.layers=len(nodes)
        self.nodes=nodes
        self.u=learn_rate # 学习率
        self.weight=list()
        self.bias=list()
        self.values=list()#层值
        self.error=0 # 误差
        self.loss=0 # 损失
        for i in range(self.layers-1):
            # 权值初始化
            self.weight.append(np.random.random((self.nodes[i],self.nodes[i+1]))-0.5)
            self.bias.append(0)

        for i in range(self.layers):
            # 节点values值初始化
            self.values.append(0)

#创建autoencoder类，可以看成是多个神经网络简单的堆叠而来
class autoencoder():
    def __init__(self):
        self.encoders=list()

    def add_one(self,nn):
        self.encoders.append(nn)

# 激活函数
def sigmod(x):
    return 1.0/(1.0+np.exp(-x))

# 前馈函数
def forwardFuction(nn,x,y):
    layers=nn.layers
    numbers=x.shape[0]
    nn.values[0]=x
    for i in range(1,layers):
        nn.values[i]=sigmod(np.dot(nn.values[i-1],nn.weight[i-1])+nn.bias[i-1])

    nn.error=y-nn.values[layers-1]
    nn.loss=1/2.0*(nn.error**2).sum()/numbers
    return nn

# 后馈函数
def backwardFunction(nn):
    layers=nn.layers
    deltas=list()
    # 初始化delta
    for i in range(layers):
        deltas.append(0)

    # 最后一层的delta为
    deltas[layers-1]=-nn.error*nn.values[layers-1]*(1-nn.values[layers-1])

    # 其他层的delta为
    for j in range(1,layers-1)[::-1]:
        deltas[j]=np.dot(deltas[j+1],nn.weight[j].T)*nn.values[j]*(1-nn.values[j])

    # 更新权值weight和偏差值biase
    for k in range(layers-1):
        nn.weight[k]-=nn.u*np.dot(nn.values[k].T,deltas[k+1])/(deltas[k+1].shape[0])
        nn.bias[k]-=nn.u*deltas[k+1]/(deltas[k+1].shape[0])

    return nn

# 对神经网络进行训练
def train(nn,x,y,iterations):
    for i in range(iterations):
        forwardFuction(nn,x,y)
        backwardFunction(nn)
    return nn

def AEtrain(ae,x,interation):
    layers=len(ae.encoders)
    for i in range(layers):
        # 单层autoencoder训练
        ae.encoders[i]=train(ae.encoders[i],x,x,interation)
        # 单层训练后，获取该autoencoder层中间值，作为下一层的训练输入
        nntemp=forwardFuction(ae.encoders[i],x,x)
        x=nntemp.values[1]
    return ae

def AEbuilder(nodes):
    layers=len(nodes)
    ae=autoencoder()
    for i in range(layers-1):
        # 训练时，我们令输入等于输出，所以每一个训练时的autoencoder层为[n1,n2,n1]形式的结构
        ae.add_one(nn([nodes[i],nodes[i+1],nodes[i]]))
    return ae


if __name__ == '__main__':
    # 测试数据，x为输入，y为输出
    x=np.array(
        [[0,0,1,0,0],[0,1,1,0,1],[1,0,0,0,1],[1,1,1,0,0],[0,1,0,1,0],[0,1,1,1,1],[0,1,0,0,1],[0,1,1,0,1],[1,1,1,1,0],
         [0,0,0,1,0]])
    y=np.array([[0],[1],[0],[1],[0],[1],[0],[1],[1],[0]])
    #弄两层autoencoder，其中5为输入的维度
    nodes=[5,3,2]
    ae=AEbuilder(nodes)
    ae=AEtrain(ae,x,6000)
    # 建立完全体的autoencoder，最后层数1为输出的特征数
    nodescomplete=np.array([5,3,2,1])
    aecomplete=nn(nodescomplete)
    # 将之前训练得到的权值赋值给完全体的autoencoder
    # 训练得到的权值，即原来的autoencoder网络中每一个autoencoder中的第一层到中间层的权值
    for i in range(len(nodescomplete)-2):
        aecomplete.weight[i]=ae.encoders[i].W[0]
    # 开始进行神经网络训练，主要就是进行微调
    aecomplete=train(aecomplete,x,y,3000)
    # 打印出最后一层的输出
    print(aecomplete.values[3])
