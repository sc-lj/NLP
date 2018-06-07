# coding:utf-8

import argparse

def argument():
    parser= argparse.ArgumentParser()

    # 语料库文件存放位置
    parser.add_argument('--corpus_txt',default='../../Dataset/new_sohu.txt',help='where is corpus',type=str)

    # 测试语料库存放地址
    parser.add_argument('--test_txt',default='../../Dataset/test.txt',help='where is test corpus',type=str)

    # 停用词语料库存放地址
    parser.add_argument('--stopfile',default='../../Dataset/stopwords/stopwords.txt',help='stop words file',type=str)

    # 处理文件后，将数据保存下来
    parser.add_argument('--target_file',default='../../Dataset/test.txt',type=str,help='after deal corpus ,this data would be saved ')

    # 是否处于训练阶段
    parser.add_argument('--is_training',default=True,type=bool,help='whether is training (default:True)')

    # 是否使用lstm模型
    parser.add_argument('--is_lstm',default=True,type=bool,help='whether is use lstm model (default:True)')

    # 是否使用双向LSTM或者双向GRU模型
    parser.add_argument('--is_bidirectional',default=True,type=bool,help='whether is use bidirectional RNN model (default:True)')

    # 每轮训练的样本数量
    parser.add_argument('--batch_size',default=128,type=int,help='the sample size of every batch train')

    cnn_parse=parser.add_argument_group('CNN argument','About CNN argument')
    # cnn模型dropout概率
    cnn_parse.add_argument('--cnn_dropout',default=0.5,help='dropout argument of CNN model')
    # cnn模型的embedding大小
    cnn_parse.add_argument('--cnn_embedding',default=500,type=int,help='embedding size of cnn model')
    # cnn模型的卷积核数量
    cnn_parse.add_argument('--cnn_filter_num',default=64,type=int,help='filter num of cnn model')
    # cnn模型卷积核大小，给定的list，是多个卷积核的大小
    cnn_parse.add_argument('--cnn_filter_size',default=[3,4,5],type=list,help='Comma-separated filter sizes (default: "3,4,5")')
    # 输入cnn模型的句子最大长度
    cnn_parse.add_argument('--cnn_maxlen',default=300,type=int,help='max sequence length of every sequence')
    # cnn模型的学习率
    cnn_parse.add_argument('--cnn_learn_rate',default=0.98,type=int,help='learning rate of cnn model')


    lstm=parser.add_argument_group('LSTM','About LSTM argument')
    # lstm模型每层的隐藏单元数
    lstm.add_argument('--rnn_hidden_unite',default=128,type=int,help='hidden state cell number of lSTM model ')
    # lstm模型的dropout 概率
    lstm.add_argument('--lstm_dropout',default=0.6,type=int,help='dropout prob of LSTM model')
    # lstm模型的层数，
    lstm.add_argument('--lstm_layer_num',default=5,type=int,help='the layer number of lstm ')

    # 是否使用多层双向lstm
    lstm.add_argument('--multbilstm',default=True,type=bool,help='whether use multiple bilstm (default: False)')

    # lstm 模型优化学习率
    lstm.add_argument('--learn_rate',default=0.9,type=int,help='the learning rate of lstm model')

    arg=parser.parse_args()
    return arg

import logging

def log_config():
    logger=logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    # sh=logging.FileHandler('./log.log')
    sh=logging.StreamHandler()
    fmt='%(asctime)s %(filename)s %(funcName)s %(lineno)s line %(levelname)s >>%(message)s'
    dtfmt='%Y-%m-%d %H:%M:%S'
    formatter=logging.Formatter(fmt=fmt,datefmt=dtfmt)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


