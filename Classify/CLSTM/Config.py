# coding:utf-8

import argparse

def argument():
    parser= argparse.ArgumentParser()
    cnn_parse=parser.add_argument_group('CNN argument','About CNN argument')
    cnn_parse.add_argument('--cnn_dropout',default=0.5,help='dropout argument of CNN model')
    cnn_parse.add_argument('--cnn_embedding',default=500,type=int,help='embedding size of cnn model')
    cnn_parse.add_argument('--cnn_filter_num',default=128,type=int,help='filter num of cnn model')
    cnn_parse.add_argument('--cnn_filter_size',default=[3,4,5],type=list,help='Comma-separated filter sizes (default: "3,4,5")')
    cnn_parse.add_argument('--cnn_seq_length',default=500,type=int,help='max sequence length of every sequence')


    lstm=parser.add_argument_group('LSTM','About LSTM argument')
    lstm.add_argument('--lstm_dropout',default=0.6)
    arg=parser.parse_args()
    return arg




















