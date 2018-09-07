# A Neural Attention Model for Sentence Summarization
本文提出的是一个基于全数据驱动文本摘要生成方法(Attention-Based Summarization (ABS))，该方法基于输入的语句采用局部注意力模型来生成摘要的每个单词。\
大部分成功的文本摘要生成方法，利用萃取法裁剪和粘贴部分文本来生成凝练的文本摘要。与此相反，抽象的文本摘要生成方法意图生成自上而下的文本摘要，即文本摘要不再是原始文本的一部分。

该模型是受到seq2seq模型的启发，encoder端模仿了Bahdanau等人的基于注意力的编码器，因为它在输入文本上学习潜在的软对齐以帮助构造摘要；decoder端是采用光束搜索方式以及在模拟萃取元素时添加了而外的特征。




