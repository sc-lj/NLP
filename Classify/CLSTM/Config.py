# coding:utf-8

import argparse

def argument():
    parser= argparse.ArgumentParser()
    cnn_parse=parser.add_argument_group('CNN argument','About CNN argument')
    # cnn模型dropout概率
    cnn_parse.add_argument('--cnn_dropout',default=0.5,help='dropout argument of CNN model')
    # cnn模型的embedding大小
    cnn_parse.add_argument('--cnn_embedding',default=500,type=int,help='embedding size of cnn model')
    # cnn模型的卷积核数量
    cnn_parse.add_argument('--cnn_filter_num',default=128,type=int,help='filter num of cnn model')
    # cnn模型卷积核大小，给定的list，是多个卷积核的大小
    cnn_parse.add_argument('--cnn_filter_size',default=[3,4,5],type=list,help='Comma-separated filter sizes (default: "3,4,5")')
    # 输入cnn模型的句子最大长度
    cnn_parse.add_argument('--cnn_maxlen',default=300,type=int,help='max sequence length of every sequence')
    # cnn模型的学习率
    cnn_parse.add_argument('--cnn_learn_rate',default=0.98,type=int,help='learning rate of cnn model')
    #


    lstm=parser.add_argument_group('LSTM','About LSTM argument')
    lstm.add_argument('--rnn_hidden_unite',default=128,type=int,help='hidden state cell number of lSTM model ')
    lstm.add_argument('--lstm_dropout',default=0.6,type=int,help='dropout prob of LSTM model')

    arg=parser.parse_args()
    return arg



