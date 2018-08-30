# coding :utf-8

import CRFPP
import os,codecs
import re,string
import subprocess



re_han_default = re.compile("([\u4E00-\u9FD5]+)", re.U)
re_skip_default = re.compile("(\r\n|\s)", re.U)
# 中文utf编码格式区间
re_han = re.compile("([^a-zA-Z0-9\u4E00-\u9FD5\r\n\s]+)", re.U)



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


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


 # 4-tags for character tagging:B(Begin),E(End),M(Middle),S(Single)
def corpus(infiles,outfile):
    f=open(infiles,'r',encoding="utf-8")
    outdata = codecs.open(outfile, 'w', 'utf-8')
    lines=f.readlines()
    for line in lines:
        line=strQ2B(line)
        sentence=re_han.sub("",line)
        for words in re.split(" |\u0020",sentence):
            if len(words.strip())==0:
                continue
            if len(words)==1:
                outdata.write(words+"\tS\n")
            else:
                #
                words=re_han_default.split(words)
                outdata.write(words[0]+"\tB\n")
                for i in range(1,len(words)-1):
                    outdata.write(words[i] + "\tM\n")
                outdata.write(words[-1]+"\tE\n")
        outdata.write("\n")

    f.close()
    outdata.close()





if __name__ == '__main__':
    # corpus("../Dataset/segment/training/pku_training.txt","../Dataset/segment/training/pku_corpus.txt")
    crf=CRFModel("./crf_model")
    crf.crf_learn("../Dataset/segment/training/pku_corpus.txt")



