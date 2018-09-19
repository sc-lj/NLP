# coding:utf-8
import tensorflow as tf


def nll_gaussian(y_pred_mean, y_pred_sd, y_test):
    ## element wise square
    square = tf.square(y_pred_mean - y_test)  ## preserve the same shape as y_pred.shape
    ms = tf.add(tf.divide(square, y_pred_sd), tf.log(y_pred_sd))
    ## axis = -1 means that we take mean across the last dimension
    ## the output keeps all but the last dimension
    ## ms = tf.reduce_mean(ms,axis=-1)
    ## return scalar
    ms = tf.reduce_mean(ms)
    return (ms)


def weight_variable(shape):
    ## weight variable, initialized with truncated normal distribution
    initial = tf.truncated_normal(shape, stddev=0.01, dtype="float32")
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.001, shape=shape, dtype="float32")
    return tf.Variable(initial)


def fully_connected_layer(h0, n_h0, n_h1, verbose=True):
    '''
    h0   :  tensor of shape (n_h0, n_h1)
    n_h0 :  scalar
    n_h1 :  scalar
    '''
    W1 = weight_variable([n_h0, n_h1])
    b1 = bias_variable([n_h1])
    h1 = tf.matmul(h0, W1) + b1
    return (h1, (W1, b1))

def mse(y_pred,y_test, verbose=True):
    '''
    y_pred : tensor
    y_test : tensor having the same shape as y_pred
    '''
    ## element wise square
    square = tf.square(y_pred - y_test)## preserve the same shape as y_pred.shape
    ## mean across the final dimensions
    ms = tf.reduce_mean(square)
    return(ms)

def define_model(n_feature, n_hs, n_output, verbose=True, NLL=True):
    x_input = tf.placeholder(tf.float32, [None, n_feature])
    y_input = tf.placeholder(tf.float32, [None, 1])

    h_previous = x_input
    n_h_previous = n_feature
    paras = []
    for ilayer, n_h in enumerate(n_hs, 1):
        h, p = fully_connected_layer(h_previous, n_h_previous, n_h, verbose)
        h_previous = tf.nn.relu(h)
        n_h_previous = n_h
        paras.append(p)
    y_mean, p = fully_connected_layer(h_previous, n_h_previous, n_output, verbose)
    paras.append(p)

    if NLL:
        if verbose:
            print("  output layer for y_sigma")
        y_sigma, p = fully_connected_layer(h_previous, n_h_previous, n_output, verbose)
        ## for numerical stability this enforce the variance to be more than 1E-4
        y_sigma = tf.clip_by_value(t=tf.exp(y_sigma),
                                   clip_value_min=tf.constant(1E-4),
                                   clip_value_max=tf.constant(1E+100))

        paras.append(p)
        loss = nll_gaussian(y_mean, y_sigma, y_input)
        y = [y_mean, y_sigma]
    else:
        loss = mse(y_mean, y_input)
        y = [y_mean]
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    inputs = [x_input, y_input]

    return (loss, train_step, inputs, y, paras)


loss, train_step, inputs, y, _ = define_model(n_feature=1,
                                              n_hs=[500, 300, 100],
                                              n_output=1)

[x_input, _] = inputs
[y_mean, y_sigma] = y

print("___" * 10)
print("loss=MSE")
## For comparison purpose, we also consider MSE as a loss
loss_mse, train_step_mse, inputs_mse, y_mse, _ = define_model(
    n_feature=1,
    n_hs=[500, 300, 100],
    n_output=1,
    NLL=False)
[x_input_mse, _] = inputs_mse
[y_mean_mse] = y_mse

from sklearn.utils import shuffle

n_epochs = 8000
n_batch = 500


def train(train_step, loss, inputs,
          x_train, y_train, n_epochs, n_batch):
    x_input, y_input = inputs

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    lvalues = []
    for count in range(n_epochs):
        x_shuffle, y_shuffle = shuffle(x_train, y_train)
        for i in range(0, x_train.shape[0], n_batch):
            batch_xs = x_shuffle[i:i + n_batch]
            batch_ys = y_shuffle[i:i + n_batch]

            sess.run(train_step,
                     feed_dict={x_input: batch_xs,
                                y_input: batch_ys})
        lv = sess.run(loss,
                      feed_dict={x_input: x_train,
                                 y_input: y_train})
        lvalues.append(lv)
        if count % 1000 == 1:
            print("  epoch={:05.0f}: {:5.3f}".format(count, lv))
    return (sess, lvalues, lv)


print("loss=NLL")
steps_per_cycle = 1
def sinfun(xs,noise=0.001):
    import random, math
    xs = xs.flatten()
    def randomNoise(x):
        ax = 2 - np.abs(x)
        wnoise = random.uniform(-noise*ax,
                                 noise*ax)
        return(math.sin(x * (2 * math.pi / steps_per_cycle) ) + wnoise)
    vec = [randomNoise(x) - x  for x in xs]
    return(np.array(vec).flatten())

inc = 0.001
import numpy as np
x_train =np.concatenate([np.arange(-2,-1.5,inc),
                         np.arange(-1,2,inc)])
x_train = x_train.reshape(len(x_train),1)
y_train0 =  sinfun(x_train,noise=0.5)
y_train = y_train0.reshape(len(y_train0),1)

sess, lvalues, lv = train(train_step, loss,
                          inputs,
                          x_train, y_train,
                          n_epochs, n_batch)

