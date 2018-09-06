#coding:utf-8

import re
import operator
import six,jieba,thulac,pyhanlp
from six.moves import range
from collections import Counter
from collections import Iterable,Generator

# 中文句子分割标点符号,其中会包含英文的标点分割符号
re_ch_sent_split=u"[?!;,，？！……。；…\n：:、“”\"\'《 》——‘’〝〞#﹟（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】\[\]`\(\)]"
# 英文句子分割标点符号
re_en_sent_split='[\[\]\n.!?,;:\t\-"\(\)\'\u2019\u2013{}<>]'

# 对短语分割
re_ch_word_split=u"[\u4E00-\u9FD5]"
re_en_word_split=u'[^a-zA-Z0-9_\+\-/]'

def is_number(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False


class Rake(object):
    def __init__(self, stop_words_path=None,min_char_length=1, max_words_length=5, min_keyword_frequency=1,
                 min_words_length_adj=1, max_words_length_adj=1, min_phrase_freq_adj=2,chinese=True):
        """
        :param stop_words_path: 停用词文件路径
        :param min_char_length:
        :param max_words_length: 词组中最多可以有多少个单词
        :param min_keyword_frequency: 关键字至少出现的次数
        :param min_words_length_adj:
        :param max_words_length_adj:
        :param min_phrase_freq_adj:
        :param ch:是否处理的是中文，默认处理中文
        """
        self.ch = chinese
        if stop_words_path:
            stop_words_file=stop_words_path
        elif not stop_words_path and chinese:
            stop_words_file = "./data/stoplists/ChineseStoplist.txt"
        else:
            stop_words_file = "./data/stoplists/SmartStoplist.txt"

        self.__stop_words_list = self.load_stop_words(stop_words_file)
        self.__min_char_length = min_char_length
        self.__max_words_length = max_words_length
        self.__min_keyword_frequency = min_keyword_frequency
        self.__min_words_length_adj = min_words_length_adj
        self.__max_words_length_adj = max_words_length_adj
        self.__min_phrase_freq_adj = min_phrase_freq_adj


    def load_stop_words(self,stop_words_file):
        """
        Utility function to load stop words from a file and return as a list of words
        @param stop_word_file Path and file name of a file containing stop words.
        @return list A list of stop words.
        """

        stop_words = []
        for line in open(stop_words_file):
            if line.strip()[0:1] != "#":
                for word in line.split():  # in case more than one per line
                    stop_words.append(word)
        return stop_words


    def split_sentences(self,text):
        """
        Utility function to return a list of sentences.
        @param text The text that must be split in to sentences.
        """
        if self.ch:
            re_compile = re_ch_sent_split
        else:
            re_compile = re_en_sent_split
        sentence_delimiters = re.compile(re_compile, re.U)
        sentences = sentence_delimiters.split(text)
        return sentences


    def separate_words(self,text, min_word_return_size):
        """
        Utility function to return a list of all words that are have a length greater than a specified number of characters.
        @param text The text that must be split in to words.
        @param min_word_return_size The minimum no of characters a word must have to be included.
        """
        if self.ch:
            word_split = re_ch_word_split
        else:
            word_split = re_en_word_split
        splitter = re.compile(word_split, re.U)
        words = []
        for single_word in splitter.split(text):
            current_word = single_word.strip().lower()
            # leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
            if len(current_word) > min_word_return_size and current_word != '' and not is_number(current_word):
                words.append(current_word)
        return words


    def calculate_word_scores(self,phraseList):
        """
        calculate individual word scores
        :param phraseList:
        :return:
        """
        word_frequency = {}
        word_degree = {}
        for phrase in phraseList:
            word_list = self.separate_words(phrase, 0)
            word_list_length = len(word_list)
            word_list_degree = word_list_length - 1
            # if word_list_degree > 3: word_list_degree = 3 #exp.
            for word in word_list:
                word_frequency.setdefault(word, 0)
                word_frequency[word] += 1
                word_degree.setdefault(word, 0)
                word_degree[word] += word_list_degree  # orig.
                # word_degree[word] += 1/(word_list_length*1.0) #exp.
        for item in word_frequency:
            word_degree[item] = word_degree[item] + word_frequency[item]

        # Calculate Word scores = deg(w)/frew(w)
        word_score = {}
        for item in word_frequency:
            word_score.setdefault(item, 0)
            word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)  # orig.
        # word_score[item] = word_frequency[item]/(word_degree[item] * 1.0) #exp.
        return word_score


    # Function that extracts the adjoined candidates from a single sentence
    def adjoined_candidates_from_sentence(self,s):
        stoplist=self.__stop_words_list
        # Initializes the candidate list to empty
        candidates = []
        # Splits the sentence to get a list of lowercase words
        sl = s.lower().split()
        # For each possible length of the adjoined candidate
        for num_keywords in range(self.__min_words_length_adj, self.__max_words_length_adj + 1):
            # Until the third-last word
            for i in range(0, len(sl) - num_keywords):
                # Position i marks the first word of the candidate. Proceeds only if it's not a stopword
                if sl[i] not in stoplist:
                    candidate = sl[i]
                    # Initializes j (the pointer to the next word) to 1
                    j = 1
                    # Initializes the word counter. This counts the non-stopwords words in the candidate
                    keyword_counter = 1
                    contains_stopword = False
                    # Until the word count reaches the maximum number of keywords or the end is reached
                    while keyword_counter < num_keywords and i + j < len(sl):
                        # Adds the next word to the candidate
                        candidate = candidate + ' ' + sl[i + j]
                        # If it's not a stopword, increase the word counter. If it is, turn on the flag
                        if sl[i + j] not in stoplist:
                            keyword_counter += 1
                        else:
                            contains_stopword = True
                        # Next position
                        j += 1
                    # Adds the candidate to the list only if:
                    # 1) it contains at least a stopword (if it doesn't it's already been considered)
                    # AND
                    # 2) the last word is not a stopword
                    # AND
                    # 3) the adjoined candidate keyphrase contains exactly the correct number of keywords (to avoid doubles)
                    if contains_stopword and candidate.split()[-1] not in stoplist and keyword_counter == num_keywords:
                        candidates.append(candidate)
        return candidates


    # Function that filters the adjoined candidates to keep only those that appears with a certain frequency
    def filter_adjoined_candidates(self,candidates):
        # Creates a dictionary where the key is the candidate and the value is the frequency of the candidate
        candidates_freq = Counter(candidates)
        filtered_candidates = []
        # Uses the dictionary to filter the candidates
        for candidate in candidates:
            freq = candidates_freq[candidate]
            if freq >= self.__min_phrase_freq_adj:
                filtered_candidates.append(candidate)
        return filtered_candidates


    # Function that extracts the adjoined candidates from a list of sentences and filters them by frequency
    def extract_adjoined_candidates(self,sentence_list):
        adjoined_candidates = []
        for s in sentence_list:
            # Extracts the candidates from each single sentence and adds them to the list
            adjoined_candidates += self.adjoined_candidates_from_sentence(s)
        # Filters the candidates and returns them
        return self.filter_adjoined_candidates(adjoined_candidates)

    def is_acceptable(self,phrase):
        """
        判断英文词组中，需要的最短字符，最大单词数，字母数必须大于数字数
        :param phrase:
        :return:
        """
        # a phrase must have a min length in characters
        if len(phrase) < self.__min_char_length:
            return 0

        # a phrase must have a max number of words
        words = phrase.split()
        if len(words) > self.__max_words_length:
            return 0

        digits = 0
        alpha = 0
        for i in range(0, len(phrase)):
            if phrase[i].isdigit():
                digits += 1
            elif phrase[i].isalpha():
                alpha += 1

        # a phrase must have at least one alpha character
        if alpha == 0:
            return 0

        # a phrase must have more alpha than digits characters
        if digits > alpha:
            return 0
        return 1


    def build_stop_word_regex(self):
        stop_word_regex_list = []
        for word in self.__stop_words_list:
            word_regex = "\\b"+word.strip()+"\\b"
            stop_word_regex_list.append(word_regex)
        stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
        return stop_word_pattern


    def generate_en_candidate_keywords(self,sentence_list):
        """
        产生英文的候选关键字
        generate candidate keyword scores
        :param sentence_list:
        :return:
        """
        stopword_pattern = self.build_stop_word_regex()
        phrase_list = []
        for s in sentence_list:
            tmp = re.sub(stopword_pattern, '|', s.strip())
            phrases = tmp.split("|")
            for phrase in phrases:
                phrase = phrase.strip().lower()
                if phrase != "" and self.is_acceptable(phrase):
                    phrase_list.append(phrase)
        phrase_list += self.extract_adjoined_candidates(sentence_list)
        return phrase_list


    def join_ch_cut_word(self,words_list):
        """
        将分词好的词表，过滤掉停用词，并以停用词为分界点，对其余词进行合并
        :param words_list:
        :return:
        """
        words=[]
        phrase=""
        if  not isinstance(words_list,Generator) and not isinstance(words_list,list):
            words_list=words_list.split(" ")
        # print(list(words_list))
        for word in words_list:
            if word not in self.__stop_words_list and re.match(re_ch_word_split,word):
                phrase+=word
            else:
                if phrase!="":
                    words.append(phrase)
                phrase=""
        return words


    def generate_ch_candidate_keywords(self,sentence_list):
        """
        产生中文的候选关键字
        :param sentence_list:
        :return:
        """
        thulac1=thulac.thulac(seg_only=True)
        phrase_list = []
        for s in sentence_list:
            cut=jieba.cut(s)
            # cut=thulac1.cut(s,text=True)
            phrase=self.join_ch_cut_word(cut)
            if len(phrase)!=0:
                phrase_list.extend(phrase)
        phrase_list+=""
        return phrase_list


    def generate_candidate_keyword_scores(self,phrase_list, word_score):
        keyword_candidates = {}
        for phrase in phrase_list:
            if self.__min_keyword_frequency > 1:
                if phrase_list.count(phrase) < self.__min_keyword_frequency:
                    continue
            keyword_candidates.setdefault(phrase, 0)
            word_list = self.separate_words(phrase, 0)
            candidate_score = 0
            for word in word_list:
                candidate_score += word_score[word]
            keyword_candidates[phrase] = candidate_score
        return keyword_candidates


    def run(self, text):
        sentence_list = self.split_sentences(text)
        print(sentence_list)
        if self.ch:
            phrase_list=self.generate_ch_candidate_keywords(sentence_list)
        else:
            phrase_list = self.generate_en_candidate_keywords(sentence_list)
        print(phrase_list)
        # word_scores = self.calculate_word_scores(phrase_list)
        #
        # keyword_candidates = self.generate_candidate_keyword_scores(phrase_list, word_scores)
        #
        # sorted_keywords = sorted(six.iteritems(keyword_candidates), key=operator.itemgetter(1), reverse=True)
        # return sorted_keywords


if __name__ == '__main__':

    en_text = u"Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types."
    ch_text="美国阿拉斯加州一架观光飞机坠毁，4人遇难1人失踪。（图源：推特）海外网8月7日电 一架观光飞机在美国阿拉斯加州德纳里国家公园坠毁，造成至少4人遇难，1人失踪。美联社、美国广播公司（ABC）等外媒7日报道称，事发于当地时间上周六（4日）晚18点左右，事发时，机上载有4名乘客及一名飞行员。报道指出，机上乘客均为波兰公民，但其姓名未透露，飞行员身份目前也已被确认。公园管理局正与波兰驻洛杉矶领事馆联系。报道指出，飞行员在失事后成功进行了两次无线电呼救，并在联系中断前通知有伤员。救援部门接到呼救电话后立即前往事发地，但由于天气条件恶劣，救援人员当天没能与飞机取得联系，也未能找到其坠落地。随着救援行动持续展开，当地时间周一（6日），相关机构派出的直升机在事故现场找到4具遇难者遗体，另外一人失踪，但推测其已丧生。目前美国国家交通安全局、旅游飞机业主公司和国家公园管理局已就此事展开调查。事实上，类似的飞行事故近日在美国时有发生。此前不久，当地时间周日（5日），一架双引擎小型飞机在美国加利福尼亚州南部一停车场坠毁，造成5人死亡。据称，飞机撞上了一辆汽车，所幸的是车主当时不在车内。坠机时地面上没有人受伤，也没有引发火灾。坠机后，警方封锁了附近几条道路以及一家购物中心。5月10日晚间8点40分左右，加州圣地亚哥斯堪的纳维亚航空学院的双引擎小飞机失联，多个目击者的报警电话证实在圣地亚哥郡的朱利安山区有飞机坠毁。警方赶赴现场后发现坠机事件引发山火，过火面积达12英亩（约5万平方米）。由于坠机地点地势险要，无法靠近，直到13日救援人员才找到飞机残骸和三具遇难者尸体。经确认残骸为失联飞机，遇难者为中国公民。（海外网 姚凯红）本文系版权作品，未经授权严禁转载。海外视野，中国立场，登陆人民日报海外版官网——海外网www.haiwainet.cn或“海客”客户端，领先一步获取权威资讯。责编：姚凯红、王栋 "

    chinese = True
    if chinese:
        text=ch_text
    else:
        text=en_text
    rake = Rake(chinese=chinese)
    rake.run(text)

