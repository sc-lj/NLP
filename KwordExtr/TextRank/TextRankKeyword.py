#-*- encoding:utf-8 -*-

import networkx as nx
import numpy as np
import os
import util
from Segment import Segmentation
class TextRankKeyword(object):
    
    def __init__(self, stop_words_file = "./stopwords.txt",
                 allow_speech_tags = util.allow_speech_tags, 
                 delimiters = util.sentence_delimiters):
        """
        Keyword arguments:
        stop_words_file  --  str，指定停止词文件路径（一行一个停止词），若为其他类型，则使用默认停止词文件
        delimiters       --  默认值是`?!;？！。；…\n`，用来将文本拆分为句子。
        
        Object Var:
        words_no_filter      --  对sentences中每个句子分词而得到的两级列表。
        words_no_stop_words  --  去掉words_no_filter中的停止词而得到的两级列表。
        words_all_filters    --  保留words_no_stop_words中指定词性的单词而得到的两级列表。
        """
        self.text = ''
        self.keywords = None
        
        self.seg = Segmentation(stop_words_file=stop_words_file, 
                                allow_speech_tags=allow_speech_tags, 
                                delimiters=delimiters)

        self.sentences = None
        self.words_no_filter = None     # 2维列表
        self.words_no_stop_words = None
        self.words_all_filters = None
        
    def analyze(self, text,
                window = 2, 
                lower = False,
                vertex_source = 'all_filters',
                edge_source = 'no_stop_words',
                pagerank_config = {'alpha': 0.85,}):
        """分析文本

        Keyword arguments:
        text       --  文本内容，字符串。
        window     --  窗口大小，int，用来构造单词之间的边。默认值为2。
        lower      --  是否将文本转换为小写。默认为False。
        vertex_source   --  选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点。
                            默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。关键词也来自`vertex_source`。
        edge_source     --  选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点之间的边。
                            默认值为`'no_stop_words'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。边的构造要结合`window`参数。
        """
        
        # self.text = util.as_text(text)
        self.text = text
        self.word_index = {}
        self.index_word = {}
        self.keywords = []
        self.graph = None
        
        result = self.seg.segment(text=text, lower=lower)
        self.sentences = result.sentences
        self.words_no_filter = result.words_no_filter
        self.words_no_stop_words = result.words_no_stop_words
        self.words_all_filters   = result.words_all_filters

        options = ['no_filter', 'no_stop_words', 'all_filters']

        if vertex_source in options:
            _vertex_source = result['words_'+vertex_source]
        else:
            _vertex_source = result['words_all_filters']

        if edge_source in options:
            _edge_source   = result['words_'+edge_source]
        else:
            _edge_source   = result['words_no_stop_words']

        self.keywords = util.sort_words(_vertex_source, _edge_source, window = window, pagerank_config = pagerank_config)

    def get_keywords(self,text=None, num = None, word_min_len = 1):
        """获取最重要的词组长度大于等于word_min_len的关键词。
        获取6个关键字
        Return:
        关键词列表。
        """
        if not self.keywords:
            self.analyze(text)
        result = []
        count = 0
        if not num:
            num=int(len(self.keywords)/3)
        for item in self.keywords:
            if count >= num:
                break
            if len(item.word) >= word_min_len:
                result.append(item)
                count += 1
        return result
    
    def get_keyphrases(self,text=None, keywords_num =None, min_occur_num = 2):
        """获取关键短语。
        获取 keywords_num 个关键词构造的可能出现的短语，要求这个短语在原文本中至少出现的次数为min_occur_num。

        Return:
        关键短语的列表。
        """
        if not self.keywords:
            self.analyze(text)
        keywords_set = set([ item.word for item in self.get_keywords(num=keywords_num, word_min_len = 1)])
        keyphrases = set()
        for sentence in self.words_no_filter:
            one = []
            for word in sentence:
                if word in keywords_set:
                    one.append(word)
                else:
                    if len(one) >  1:
                        keyphrases.add(''.join(one))
                    if len(one) == 0:
                        continue
                    else:
                        one = []
            # 兜底
            if len(one) >  1:
                keyphrases.add(''.join(one))

        return [phrase for phrase in keyphrases 
                if self.text.count(phrase) >= min_occur_num]

if __name__ == '__main__':
    content="""阿的江：八一需重新定位 我们有机会但该进的没进   新浪体育讯12月24日，回到主场的北京金隅迎战八一，四节苦战之后，北京队凭借出色表现以84比78险胜八一双鹿，赢取五连胜。赛后，八一双鹿队主教练阿的江点评比赛，“首先感谢媒体朋友们在平安夜来到篮球场观看比赛，气氛非常好，跟过节一样。这场比赛，是八一双鹿队开赛以来打得比较紧的一场比赛，无论是攻防成功率都非常低，特别是进攻， 许多该进的球都不进。做为我们来讲，回到北京想把比赛打好，却给自己背上一些包袱，毕竟联赛才刚刚开始，通过这场比赛，八一双鹿队要重新定位，八一双鹿队现在只是爬坡期，没有成为强队，如果认为自己赢了几场球就给自己造成压力，没有必要。现在是比赛初期，打了几场好球，但赢球之后 反而不能及时调整摆正位置的话，还会出现问题。漫长的联赛会出现各种各样的问题，祝贺北京队获得胜利，北京队每场比赛打得都非常有激情。 祝大家圣诞快乐。”问：北京队取得五连胜，请点评一下北京队的表现。阿的江：这场比赛的比分是84比78，说实话，两队得分都不高，但是八一双鹿队表现得比北京队更差，所以北京队获得胜利。现在北京队开赛取得五连 胜，是一个非常好的开局，我希望他们不要犯我们这场比赛的错误，能够以平常心来对待这些比赛的结果，胜利来之不易，但保持更难。我们也会认真准备。问：你觉得北京队赢在哪里？阿的江：北京队在胶着状态中把握住了机会，八一双鹿队同样也有机会，但该进的都没进，特别是在我们反超比赛的时候，防守出现一些漏洞，北京队及时抓住机会，北京队相对来说，抓机会比较合理。问：前三节都是平局，谈谈第四节吧，八一双鹿队好像失误多，命中率很低。阿的江：今天是八一双鹿队开赛以来得分最低的，不过第四节输给双外援这是第二次，实际上我们在第四节对抗双外援方面有进步，有改观。有些队员像大郅的状态起伏，比赛太紧，不能强求所有队员每场都表现很好，希望 他们逐步恢复状态，打、调相结合，今天的命中率不是他正常的命中率，非常差，这与他今年参加两个大赛，亚运会结束后就投入联赛有关系，作为他来讲也必须及时调整。其实今天的问题我们就是出在命中率上。这种比赛也少见，三节都打平，看得出两家在圣诞夜给球迷奉献了一场精彩的比赛。八一双鹿队队员许钟豪表示，“我们所有的队员都想打好这场比赛，确实给自 己背上一些包袱压力，接下来还有更多的比赛，我们还是脚踏实地重新开 始吧。”(小三儿)"""
    keyword=TextRankKeyword(stop_words_file="./stopwords.txt")
    keyword.analyze(content)
    a=keyword.get_keywords()
    print(a)

