# coding:utf-8

# 限制性玻尔兹曼机
import numpy as np


class RBMs():
    def __init__(self,num_visible,num_hidden):
        self.num_visible=num_visible
        self.num_hidden=num_hidden

        # 定义一个伪随机数的种子，只要该随机数相同，产生的随机数序列都是相同的.
        np_rng=np.random.RandomState(1234)
        self.weight=np.asarray(np_rng.uniform(low=-0.1*np.sqrt(6./(num_hidden+num_visible)),
                                              high=0.1*np.sqrt(6./(num_visible+num_hidden)),
                                              size=(num_visible,num_hidden)))
        self.weight=np.insert(self.weight,0,0,axis=0)
        self.weight=np.insert(self.weight,0,0,axis=1)

    def _logistic(self,x):
        return 1.0/(1+np.exp(-x))

    def train(self,data,max_epochs=1000,learn_rate=0.1):
        num_examples=data.shape[0]
        # 第一列插入值为1的偏差
        data=np.insert(data,0,1,axis=1)
        for epoch in range(max_epochs):
            pos_hidden_activations=np.dot(data,self.weight)
            pos_hidden_probs=self._logistic(pos_hidden_activations)
            pos_hidden_probs[:,0]=1 # 固定偏差
            pos_hidden_states=pos_hidden_probs>np.random.rand(num_examples,self.num_hidden+1)
            pos_associations=np.dot(data.T,pos_hidden_probs)

            neg_visible_activations=np.dot(pos_hidden_states,self.weight.T)
            neg_visible_probs=self._logistic(neg_visible_activations)
            neg_visible_probs[:,0]=1
            neg_hidden_activation=np.dot(neg_visible_probs,self.weight)
            neg_hidden_probs=self._logistic(neg_hidden_activation)
            neg_associations=np.dot(neg_visible_probs.T,neg_hidden_probs)

            self.weight+=learn_rate*((pos_associations-neg_associations)/num_examples)
            error=np.sum((data-neg_visible_probs)**2)

    def run_visible(self,data):
        num_examples=data.shape[0]
        hidden_states=np.ones((num_examples,self.num_hidden+1))
        data=np.insert(data,0,1,axis=1)

        hidden_activations=np.dot(data,self.weight)
        hidden_probs=self._logistic(hidden_activations)
        hidden_states[:,:]=hidden_probs>np.random.rand(num_examples,self.num_hidden+1)
        hidden_states=hidden_states[:,1:]
        return hidden_states

    def daydream(self,num_samples):
        samples=np.ones((num_samples,self.num_visible+1))
        samples[0,1:]=np.random.rand(self.num_visible)
        for i in range(1,num_samples):
            visible=samples[i-1,:]
            hidden_activations=np.dot(visible,self.weight)
            hidden_probs=self._logistic(hidden_activations)
            hidden_states=hidden_probs>np.random.rand(self.num_hidden+1)
            hidden_states[0]=1

            visible_activations=np.dot(hidden_states,self.weight.T)
            visible_probs=self._logistic(visible_activations)
            visible_states=visible_probs>np.random.rand(self.num_visible+1)
            samples[i,:]=visible_states

        return samples[:,1:]



"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""
import timeit

try:
    import PIL.Image as Image
except ImportError:
    import Image

import theano
import theano.tensor as T
import os

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import tile_raster_images
from MLP import load_data


# start-snippet-1
class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(self,input=None,n_visible=784, n_hidden=500,W=None,hbias=None,vbias=None,numpy_rng=None,theano_rng=None):
        """
        :param input:
        :param n_visible:可视单元的数量、
        :param n_hidden: 隐藏单元的数量
        :param W: 对单个RBMs为None或者RBM作为深度信念网络的一部分使用共享权重。在深度信念网络中，权重在RBMs和MLP层之间是共享的。
        :param hbias:
        :param vbias:
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            initial_W = np.asarray(numpy_rng.uniform(low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                                                     high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                                                     size=(n_visible, n_hidden)),
                                   dtype=theano.config.floatX)

            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            hbias = theano.shared( value=np.zeros(n_hidden,dtype=theano.config.floatX),name='hbias',borrow=True)

        if vbias is None:
            vbias = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        # 初始化单个RBM的输入层或者DBN的第0层
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng

        self.params = [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        """ 计算能量函数 """
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        """"""
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        '''给定可视层状态推定隐藏层状态'''
        # 给定可视层样本，计算隐藏层单元的激活值
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # 给定可视层的激活值，进行gibbs抽样
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,n=1, p=h1_mean,dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''将隐藏单元激活向下传播到可视单元'''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' 给定隐藏层状态，推定可视层状态 '''
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        #  给定隐藏层的激活值，进行gibbs抽样
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,n=1, p=v1_mean,dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' 从隐藏层状态开始，进行一步GIbbs采样 '''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """ 对比散度或者持续对比散度
        :param lr:
        :param persistent: 如果选择对比散度，为None；如果为持续对比散度，共享变量需要保留gibbs链的旧状态，其shape为(batch size, number of hidden units)
        :param k: 一步Gibbs采样的数量
        """

        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        if persistent is None:
            # 对于对比散度，使用新生成的隐藏样本
            chain_start = ph_sample
        else:
            # 对于持续对比散度，使用马尔科夫链的旧状态
            chain_start = persistent

        # 执行k次抽样，scan返回的是整个Gibbs抽样链
        ([
             pre_sigmoid_nvs,
             nv_means,
             nv_samples,
             pre_sigmoid_nhs,
             nh_means,
             nh_samples
         ],
         updates) = theano.scan(
            fn=self.gibbs_hvh,
            # 给定要循环计算的输出的初始状态,与此同时也给gibbs_hvh传入了一个初始参数
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k,
            name="gibbs_hvh"
        )

        # 只需要链式抽样的最后一组
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
        # 不能对gibbs采样计算梯度，所以用consider_constant加以限制
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        for gparam, param in zip(gparams, self.params):
            # 确保学习率格式正确；更新参数
            updates[param] = param - gparam * T.cast(lr,dtype=theano.config.floatX)
        if persistent:
            updates[persistent] = nh_samples[-1]
            # 对持续对比散度使用 pseudo 似然求解
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # 对对比散度使用 cross-entropy 求解
            monitoring_cost = self.get_reconstruction_cost(updates,pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # 将输入值约等于邻近的整数值
        xi = T.round(self.input)

        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error
        这个函数需要预定义的sigmoid激活函数作为输入，优化的表达式是一种softplus的公式 log(sigmoid(x))。我们还需要引入
        交叉熵函数，这是因为当x大于30，sigmoid结果趋近于1，或者小于-30的时候，sigmoid结果趋近于0，因此我们会得到负无穷或者空值。

        之所以需要预定一个sigmoid函数作为输入，这是因为theano会使用log(scan(..))代替log(sigmoid(..))，但是这样的不会得到
        我们想要的结果。我们不能在scan中使用sigmoid，所以最有效的方法是预定义一个sigmoid函数。
        """
        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )
        return cross_entropy


def test_rbm(learning_rate=0.1, training_epochs=15,
             dataset='mnist.pkl.gz', batch_size=20,
             n_chains=20, n_samples=10, output_folder='rbm_plots',
             n_hidden=500):
    """
    This is demonstrated on MNIST.
    Demonstrate how to train and afterwards sample from it using Theano.
    :param learning_rate: learning rate used for training the RBM
    :param training_epochs: number of epochs used for training
    :param dataset: path the the pickled dataset
    :param batch_size: size of a batch used to train the RBM
    :param n_chains: number of parallel Gibbs chains to be used for sampling
    :param n_samples: number of samples to plot for each chain
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden layer of chain)
    persistent_chain = theano.shared(np.zeros((batch_size, n_hidden),dtype=theano.config.floatX),borrow=True)

    rbm = RBM(input=x, n_visible=28 * 28,n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate,persistent=persistent_chain, k=15)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = timeit.default_timer()

    for epoch in range(training_epochs):
        mean_cost = []
        for batch_index in range(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print('Training epoch %d, cost is ' % epoch, np.mean(mean_cost))

        plotting_start = timeit.default_timer()

        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))

    # Sampling from the RBM, find out the number of test samples
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        np.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )

    plot_every = 1000
    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every,
        name="gibbs_vhv"
    )

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    image_data = np.zeros(
        (29 * n_samples + 1, 29 * n_chains - 1),
        dtype='uint8'
    )
    for idx in range(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print(' ... plotting sample %d' % idx)
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    # construct image
    image = Image.fromarray(image_data)
    image.save('samples.png')
    # end-snippet-7
    os.chdir('../')



if __name__ == '__main__':
    r=RBMs(num_visible=6,num_hidden=2)
    training_data=np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0],[0,0,1,1,0,0],[0,0,1,1,1,0]])
    # r.train(training_data,max_epochs=5000)
    # print(r.weight)
    # user=np.array([[0,0,0,1,1,0]])
    # print(r.run_visible(user))































