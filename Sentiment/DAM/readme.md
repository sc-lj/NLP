# 情感分析

## 数据集

知网情感分析用词语集[本地地址](NLP/datasets/sentiment/hownet),点击下载[地址](http://www.keenage.com/html/c_bulletin_2007.htm)。

台湾大学简体中文情感极性词典[本地地址](NLP/datasets/sentiment/ntusd)。

程度副词[本地地址](NLP/datasets/sentiment/advdegree)

大部分词汇是从[github](https://github.com/data-science-lab/sentimentCN/tree/master/dict)上下载的。

# 基于双重注意力模型的微博情感分析方法

该方法首先利用现有的情感知识库构建了一个包含**情感词**、**程度副词**、**否定词**、**微博表情符号**
和**常用网络用语的微博情感符号库**;然后，采用**双向长短记忆网络模型和全连接网络**，
分别对**微博文本和文本中包含的情感符号**进行编码;接着，采用**注意力模型**分别构建微博**文本和情感符号**的语义表示，
并将两者的语义表示进行融合，以构建微博文本的最终语义表示;最后，基于所构建的语义表示对情感分类模型进行训练。



