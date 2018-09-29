## 实体关系抽取 entity relation extraction 文献阅读总结
这篇文章[《CIPS青工委学术专栏第3期 | 基于深度学习的关系抽取》](http://www.cipsc.org.cn/qngw/?p=890)有关实体关系抽取总结的很好。

传统的关系抽取方法总结：
- 基于句法解析增强的方法，Milleret al. 2000
- 基于逻辑回归的方法，Kambhatla 2004
- 基于核函数的方法，Zhao and Grishman 2005; Bunescu and Mooney 2006
- 基于条件随机场的方法,Culotta et al. 2006
- 远程监督,Distant supervision,Mintz et al. 2009
- 基于无向图模型的关系抽取方法,Yao et al. 2010
- 增强远程监督的假设,Riedel et al. 2010
- 改进实体对齐,Takamatsuet al. 2012
- 多实例多标签,Hoffmann etal. 2011
- 多标签多实例+贝叶斯网络,Surdeanu etal. 2012
- 基于深度学习的关系抽取方法(2015年之前的，CNN刚刚火起来)：


**RNN，Socher et al. 2012**

Socher et al. 2012  [Semantic compositionality through recursive matrix-vector spaces.](https://ai.stanford.edu/~ang/papers/emnlp12-SemanticCompositionalityRecursiveMatrixVectorSpaces.pdf) 提出使用递归神经网络来解决关系抽取问题。该方法首先对句子进行句法解析，然后为句法树上的每个节点学习向量表示。通过递归神经网络，可以从句法树最低端的词向量开始，按照句子的句法结构迭代合并，最终得到该句子的向量表示，并用于关系分类。该方法能够有效地考虑句子的句法结构信息，但同时该方法无法很好地考虑两个实体在句子中的位置和语义信息。

**CNN，Zeng et al. 2014**

他们采用词汇向量和词的位置向量作为卷积神经网络的输入，通过卷积层、池化层和非线性层得到句子表示。通过考虑实体的位置向量和其他相关的词汇特征，句子中的实体信息能够被较好地考虑到关系抽取中。

**CNN，新的损失函数,Santos et al. 2015**

后来，[Santos et al. 2015]还提出了一种新的卷积神经网络进行关系抽取，其中采用了新的损失函数，能够有效地提高不同关系类别之间的区分性。

**CNN，扩展至远程监督，Zeng et al. 2015**

理解远程监督 a glance at Distant Supervision

什么是远程监督呢？一开始是因为觉得人工标注数据比较费时费力，那么就有人想来个自动标注的方法。远程监督就是干这个事儿的。

假设知识库KB当中存在实体与实体的关系，那么将KB当中的关系引入到正常的自然语言句子当中进行训练，例如‘苹果’和’乔布斯’在KB中的关系是CEO，那么我们就假设类似于“【乔布斯】发布了【苹果】的新一代手机”的句子上存在CEO的关系，如此，利用KB对海量的文本数据进行自动标注，得到标注好的数据（正项），再加入一些负项，随后训练一个分类器，每个分类是一个关系，由此实现关系抽取。

09年的文章就是这个思想：在KB中有一个triplet，那么在corpus中凡是有这个entity pair的sentence全都当成含有这个relation的instance

论文总结 paper reading

**PCNN**

论文名称：[Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks](http://aclweb.org/anthology/D/D15/D15-1203.pdf)

论文内容：非常经典的文章，分段式的max pooling。后面做的文章都要引用这篇文章。

**BRCNN**

论文名称：[Bidirectional Recurrent Convolutional Neural Network for Relation Classification](http://www.aclweb.org/anthology/P/P16/P16-1072.pdf)

论文内容：本文提出了一个基于最短依赖路径（SDP）的深度学习关系分类模型，文中称为双向递归卷积神经网络模型（BRCNN）

**BiLSTM SPTree**

论文名称：[End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures](http://www.aclweb.org/anthology/P/P16/P16-1105.pdf)

论文内容：用了一种树形的结构

**BLSTM + ATT**
论文名称：[Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](http://www.aclweb.org/anthology/P16-2034)

论文内容：简单有效。使用BLSTM对句子建模，并使用word级别的attention机制。

**CNN+ATT / PCNN+ATT**

论文名称：[Neural Relation Extraction with Selective Attention over Instances](http://www.aclweb.org/anthology/P16-1200)

论文内容：使用CNN/PCNN作为sentence encoder, 并使用句子级别的attention机制。近几年标杆的存在，国内外新论文都要把它拖出来吊打一遍。

**MNRE**

论文名称：NUERAL RELATION EXTRACTION WITH MULTI-LINGUAL ATTENTION

论文内容：很有意思也很有用。单语言语料的信息如果不够，就要用到多语言的语料。NLP任务中多语言之间的信息利用是今年研究比较多的一个。不过实际做起来难度是比较大的，最主要原因还是数据比较难以采集。本文使用
(P)CNN+ATT(上面那篇)扩展到多语言语料库上，利用[多语言之间的信息](https://zhuanlan.zhihu.com/p/29970617)。性能提升比较客观。应该也只有一些大公司才有能力将这种算法落地使用。

**ResCNN-9**

论文名称：Deep Residual Learning forWeakly-Supervised Relation Extraction

论文内容：本文使用浅层（9）ResNet作为sentence encoder, 在不使用piecewise pooling 或者attention机制的情况下，性能和PCNN+ATT 接近。这就证明使用更fancy的CNN网络作为sentence encoder完全是有可能有用的。不光光可以在本任务中验证，其他的NLP任务同样可以使用。本文在github上有源代码，强烈推荐。我写的知乎笔记： https://zhuanlan.zhihu.com/p/31689694。 顺带一提的是，本文的工程实现还存在可以改进的地方。

**REPEL**

论文名称：Overcoming Limited Supervision in Relation Extraction: A Pa‚ttern-enhanced Distributional Representation Approach

论文内容：这篇文章思路比较有意思，非常值得一看。没有用深度学习，而是两个朴素的模型互相迭代，运用了半监督学习的思想。不过没有代码，如果实验结果可以复现，那么意义还是比较大的。https://zhuanlan.zhihu.com/p/32364723。

**Graph LSTM**

论文名称：Cross-Sentence N-ary Relation Extraction with Graph LSTMs

论文内容：这个就是提出了一种图形LSTM结构，本质上还是利用了SDP等可以利用的图形信息。别的部分没有什么特别的。https://zhuanlan.zhihu.com/p/32541447

**APCNNs(PCNN + ATT) + D**

论文名称：Distant Supervision for Relation Extraction with Sentence-Level Attention and Entity Descriptions

论文内容：引入实体描述信息，个人认为没什么亮点，引入外部信息固然有效，但是很多时候实际问题中遇到的实体大多是找不到实体描述信息的。 https://zhuanlan.zhihu.com/p/35051652

**PE + REINF**

论文名称：Large Scaled Relation Extraction with Reinforcement Learning

论文内容：提出强化学习用于RE任务，个人感觉挺牵强的，效果也很一般。文中提到的PE不知道是不是我代码写错了，试出来就是没什么用。  https://zhuanlan.zhihu.com/p/34811735

**CNN + ATT + TM**

论文名称： Learning with Noise: Enhance Distantly Supervised Relation Extraction with Dynamic Transition Matrix

论文内容：文章出发点很好。既然远程监督数据集最大的问题在于噪音非常之多，那么对于噪音进行描述则是非常有意义的。本文创新点有两个。第一个是，我们让模型先学习从输入空间到真实标签空间的映射，再用一个转移矩阵学习从真实标签空间到数据集标签空间的错误转移概率矩阵。这不是本文提出的方法，本文在此基础之上进行改进，将该矩阵从全局共享转化为跟输入相关的矩阵，也就是文中提到的动态转移矩阵，性能有提升。第二个出创新点在于使用了课程学习。课程学习的出发点在于模型如果先学习简单样本再学习难样本，这样一种先易后难的学习方式比随机顺序学习更好。最终在NYT数据集上有小小的提升，但是本文的思路非常值得借鉴。可只可惜没有源代码。建议读博的大佬们尝试一下，我觉得很好玩。 https://zhuanlan.zhihu.com/p/36527644



论文名称： Effectively Combining RNN and CNN for Relation Classification and Extraction

论文内容：这是一篇打比赛的文章，工程性的内容很多。核心技巧在于使用CNN, RNN模型集成。文中还提到了多种方法，不择手段提升最终模型的性能。虽然该模型训练速度可以说是非常慢了，但是还是有很多地方可以借鉴。 https://zhuanlan.zhihu.com/p/35845948



以上介绍的是关系抽取，建立在NER的基础上，接下来讨论joint模型。联合抽取个人认为是比较难做的一个任务。

**CoType**

论文名称：CoType: Joint Extraction of Typed Entities and Relations with Knowledge Bases

论文内容：坦白地说没有太看懂。才疏学浅。 https://zhuanlan.zhihu.com/p/23635696

github有源代码： https://github.com/shanzhenren/CoType

**LSTM-CRF, LSTM-LSTM,LSTM-LSTM-Bias**

论文名称： Joint Extractions of Entities and Relations Based on a Novel Tagging Scheme

论文内容：把关系抽取内容转换成序列标注任务 https://zhuanlan.zhihu.com/p/31003123

