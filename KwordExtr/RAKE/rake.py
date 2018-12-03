#coding:utf-8

import re
import operator
import six,jieba,thulac,pyhanlp
from six.moves import range
from collections import Counter,defaultdict
from collections import Iterable,Generator
import numpy as np

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
                 min_words_length_adj=1, max_words_length_adj=1, min_phrase_freq_adj=2,chinese=True,is_jieba=True):
        """
        :param stop_words_path: 停用词文件路径
        :param min_char_length:
        :param max_words_length: 词组中最多可以有多少个单词
        :param min_keyword_frequency: 关键字至少出现的次数
        :param min_words_length_adj: 词最短的毗邻距离
        :param max_words_length_adj: 词最大的毗邻距离
        :param min_phrase_freq_adj: 相同的毗邻词至少出现的次数，以过滤掉较少的
        :param ch: 是否处理的是中文，默认处理中文
        :param is_jieba: 本文提供了两种中文分词器，默认的是jieba，另一种是清华大学提供的thulac分词器
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
        thulac1 = thulac.thulac(seg_only=True)
        self.cut_word=lambda word:jieba.cut(word) if is_jieba else thulac1.cut(word,text=True)


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
            words_list=self.cut_word(text)
        else:
            splitter = re.compile(re_en_word_split, re.U)
            words_list=splitter.split(text)
            words_list=[word.strip().lower() for word in words_list]
        words = []
        for single_word in words_list:
            # leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
            if len(single_word) > min_word_return_size and single_word != '' and not is_number(single_word):
                words.append(single_word)
        return words


    def calculate_word_scores(self,phraseList):
        """
        calculate individual word scores
        :param phraseList:
        :return:
        """
        phraselist=[]
        phrases=[]
        for phrase in phraseList:
            word_list = self.separate_words(phrase, 0)
            phraselist.extend(word_list)
            phrases.append(word_list)
        phraselist=list(set(phraselist))
        phraselen = len(phraselist)
        phrasearray = np.zeros((phraselen, phraselen))

        for phrase in phrases:
            if len(phrase)==1:
                index=phraselist.index(phrase[0])
                phrasearray[index,index]+=1
            else:
                for i in range(len(phrase)-1):
                    index1 =phraselist.index(phrase[i])
                    phrasearray[index1, index1] += 1
                    for j in range(i+1,len(phrase)):
                        index2 = phraselist.index(phrase[j])
                        phrasearray[index1, index2] += 1
                        phrasearray[index2, index1] += 1
                lastword=phraselist.index(phrase[-1])
                phrasearray[lastword, lastword] += 1

        # Calculate Word scores = deg(w)/frew(w)
        word_score = {}
        for i in range(phraselen):
            deg=sum(phrasearray[i])
            freq=phrasearray[i,i]
            word_score[phraselist[i]] = deg / freq
        return word_score


    def adjoined_candidates_from_sentence(self,s):
        """
         Function that extracts the adjoined candidates from a single sentence
        :param s:
        :return:
        """
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


    def filter_adjoined_candidates(self,candidates):
        """
        Function that filters the adjoined candidates to keep only those that appears with a certain frequency
        :param candidates:
        :return:
        """
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
        if phrase != "":
            words.append(phrase)
        return words


    def generate_ch_candidate_keywords(self,sentence_list):
        """
        产生中文的候选关键字
        :param sentence_list:
        :return:
        """
        phrase_list = []
        for s in sentence_list:
            cut=self.cut_word(s)
            phrase=self.join_ch_cut_word(cut)
            if len(phrase)!=0:
                phrase_list.extend(phrase)
        return phrase_list


    def generate_candidate_keyword_scores(self,phrase_list, word_score):
        keyword_candidates = defaultdict(int)
        for phrase in phrase_list:
            if self.__min_keyword_frequency > 1:
                if phrase_list.count(phrase) < self.__min_keyword_frequency:
                    continue

            word_list = self.separate_words(phrase, 0)
            candidate_score = 0
            for word in word_list:
                candidate_score += word_score[word]
            keyword_candidates[phrase] = candidate_score
        return keyword_candidates


    def run(self, text):
        sentence_list = self.split_sentences(text)
        if self.ch:
            phrase_list=self.generate_ch_candidate_keywords(sentence_list)
        else:
            phrase_list = self.generate_en_candidate_keywords(sentence_list)
        word_scores = self.calculate_word_scores(phrase_list)
        keyword_candidates = self.generate_candidate_keyword_scores(phrase_list, word_scores)
        sorted_keywords = sorted(six.iteritems(keyword_candidates), key=operator.itemgetter(1), reverse=True)
        return sorted_keywords


if __name__ == '__main__':

    en_text = u"Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types."
    # ch_text="一架观光飞机在美国阿拉斯加州德纳里国家公园坠毁，造成至少4人遇难，1人失踪。美联社、美国广播公司（ABC）等外媒7日报道称，事发于当地时间上周六（4日）晚18点左右，事发时，机上载有4名乘客及一名飞行员。报道指出，机上乘客均为波兰公民，但其姓名未透露，飞行员身份目前也已被确认。公园管理局正与波兰驻洛杉矶领事馆联系。报道指出，飞行员在失事后成功进行了两次无线电呼救，并在联系中断前通知有伤员。救援部门接到呼救电话后立即前往事发地，但由于天气条件恶劣，救援人员当天没能与飞机取得联系，也未能找到其坠落地。随着救援行动持续展开，当地时间周一（6日），相关机构派出的直升机在事故现场找到4具遇难者遗体，另外一人失踪，但推测其已丧生。目前美国国家交通安全局、旅游飞机业主公司和国家公园管理局已就此事展开调查。事实上，类似的飞行事故近日在美国时有发生。此前不久，当地时间周日（5日），一架双引擎小型飞机在美国加利福尼亚州南部一停车场坠毁，造成5人死亡。据称，飞机撞上了一辆汽车，所幸的是车主当时不在车内。坠机时地面上没有人受伤，也没有引发火灾。坠机后，警方封锁了附近几条道路以及一家购物中心。5月10日晚间8点40分左右，加州圣地亚哥斯堪的纳维亚航空学院的双引擎小飞机失联，多个目击者的报警电话证实在圣地亚哥郡的朱利安山区有飞机坠毁。警方赶赴现场后发现坠机事件引发山火，过火面积达12英亩（约5万平方米）。由于坠机地点地势险要，无法靠近，直到13日救援人员才找到飞机残骸和三具遇难者尸体。经确认残骸为失联飞机，遇难者为中国公民。"

    ch_text = """据路透社9月6日报道称，两名消息人士向路透社透露称，一艘英国皇家海军的军舰在前往越南的路途中靠近中国南海海域的岛礁行驶，这被认为是英国皇家海军主张“航行自由”权利。文章称，几天前载有皇家海军陆战队员的排水量22000吨的海神之子号两栖舰从西沙群岛经过，这艘军舰当时行驶的目标是越南的胡志明市。在完成了在日本周边地区的部署后，从本周一开始它停靠在那里。消息人士称，当时北京派出一艘护卫舰和两架直升机对英国船只进行监控，但双方在遭遇期间保持冷静。另一个消息来源透露，海神之子号两栖舰并没有进入西沙群岛任何岛礁的12海里领海。路透社称，英国皇家海军发言人说：“海神之子号两栖舰在完全符合国际法和规范的情况下行使了航行自由权，”消息人士称，英国海军的军舰此前曾靠近南沙群岛的岛礁行驶，但没有进入岛礁12海里的领海范围内。
    　　在今年8月的中国国防部例行记者会上，国防部新闻发言人吴谦在接受采访时表示南海诸岛自古以来就是中国领土，这是一个事实。南海的航行自由没有问题，这是一个事实。南海行为准则磋商近期取得重大进展，这也是一个事实。一段时间以来，美方炒作南海问题，试图把影响航行自由的帽子扣在中方头上。我必须指出，谎言重复千遍也成为不了真理。
    　　注：英国海神之子级船坞登陆舰，于1998年5月23日由维克斯船舶工程公司开工建造，2001年3月9日下水，2003年6月19日服役。该舰满载排水量18400吨，舰长176米，宽28.9米。主要负责运送部队以及其武器、军备和一定数目的补给品作远征，抵达后使用登陆艇和直升机等工具，将部队和装备送上岸作战，亦可担任舰队旗舰，负责指挥整场两栖作战"""

    chinese = True
    if chinese:
        text=ch_text
    else:
        text=en_text
    rake = Rake(chinese=chinese)
    a=rake.run(text)
    print(a)

