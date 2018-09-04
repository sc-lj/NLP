# coding :utf-8

import CRFPP
import os,codecs
import re,string
import subprocess

class CRFModel(object):
    def __init__(self, model='model_name'):
        """
        函数说明: 类初始化
        :param model: 模型名称
        """
        self.model = model

    def add_tagger(self, tag_data):
        """
        函数说明: 添加语料
        :param tag_data: 数据
        :return:
        """
        word_str = tag_data.strip()
        if not os.path.exists(self.model):
            print('模型不存在,请确认模型路径是否正确!')
            exit()
        tagger = CRFPP.Tagger("-m {} -v 3 -n2".format(self.model))
        tagger.clear()
        for word in word_str:
            tagger.add(word)
        tagger.parse()
        return tagger

    def text_mark(self, tag_data, begin='B', middle='I', end='E', single='S'):
        """
        文本标记
        :param tag_data: 数据
        :param begin: 开始标记
        :param middle: 中间标记
        :param end: 结束标记
        :param single: 单字结束标记
        :return result: 标记列表
        """
        tagger = self.add_tagger(tag_data)
        size = tagger.size()
        tag_text = ""
        for i in range(0, size):
            word, tag = tagger.x(i, 0), tagger.y2(i)
            if tag in [begin, middle]:
                tag_text += word
            elif tag in [end, single]:
                tag_text += word + "*&*"
        result = tag_text.split('*&*')
        result.pop()
        return result

    def crf_test(self, tag_data, separator='_'):
        """
        函数说明: crf测试
        :param tag_data:
        :param separator:
        :return:
        """
        result = self.text_mark(tag_data)
        data = separator.join(result)
        return data

    def crf_learn(self, filename):
        """
        函数说明: 训练模型
        :param filename: 已标注数据源
        :return:
        """
        """
        参数解释:
        -f, –freq=INT 使用属性的出现次数不少于INT(默认为1)
        -m, –maxiter=INT 设置INT,为LBFGS的最大迭代次数 (默认10k)
        -c, –cost=FLOAT 设置FLOAT,为代价参数，过大会过度拟合 (默认1.0)
        -e, –eta=FLOAT 设置终止标准FLOAT(默认0.0001)
        -C, –convert 将文本模式转为二进制模式
        -t, –textmodel 为调试建立文本模型文件
        -a, –algorithm=(CRF-L2/CRF-L1/MIRA) 选择训练算法，默认为CRF-L2
        -p, –thread=INT 线程数(默认1)，利用多个CPU减少训练时间
        -H, –shrinking-size=INT 设置INT为最适宜的跌代变量次数 (默认20)
        template是生成特征函数的规则
        """
        crf_bash = "crf_learn -f 3 -c 4.0 template {} {}".format(filename, self.model)
        process = subprocess.Popen(crf_bash.split(), stdout=subprocess.PIPE)
        output = process.communicate()[0]
        print(output.decode(encoding='utf-8'))


if __name__ == '__main__':
    crf=CRFModel("./crf_model")
    crf.crf_learn("./pku_corpus.txt")



