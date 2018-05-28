# cnn 在文本分类上的应用

本文主要是介绍两种CNN模型在文本分类领域上的应用。
- seq-CNN
- bow-CNN

一般情况下，做自然语言处理时，对于输入词向量有两种方式，一种是直接利用训练好的词向量作为输入，比如用word2vec或者Glove训练出来的词向量；
一种是利用one-hot作为词向量。本文就是利用one-hot表示词向量模型。

对于词汇表V = { “don’t”, “hate”, “I”, “it”, “love” }，目标短语：D=“I love it”。

## seq-CNN

seq-CNN词向量的构成是将相邻的两个或几个词的one-hot组合成一个
词向量，这样相当于利用了词序信息。至于利用几个相邻的词，可以
事先确定。本文是用p来表示利用几个词的。
假设p=2。
则利用上面的词汇表和目标短语：

![seq-cnn 词向量][1]

上述词向量的维度为p|V|维，stride=1。

缺点：
这样做有个明显的缺点就是如果词汇表很大，且p也很大时，那么就会造成维度灾难，而且向量也是稀疏的。

## bow-CNN
bow-CNN模型改善了seq-CNN模型，它直接将相邻几个词的位置写入一个one-hot中。这样向量的维度还是V维，同时向量的稀疏问题得到了改善。

![bow-cnn 词向量][2]

对于传统的cnn都有多层卷积层和pooling层，但是本文提出了平行卷积层的概念，即对于同一输入向量，采用不同的卷积大小进行卷积，最后在全连接层进行合并进行softmax操作。

对于在词汇表没有的词汇，我们将会用零表示。

由于文本中存在全角数字，而全角数字的Unicode编码范围为：\uff10-\uff19 表示0-9

[1]: images/seq-cnn.jpg
[2]: images/bow-cnn.jpg