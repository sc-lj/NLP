# coding :utf-8

import CRFPP
import os,codecs
import re,string
import jieba



re_han_default = re.compile("([\u4E00-\u9FD5]+)", re.U)
re_skip_default = re.compile("(\r\n|\s)", re.U)
# 中文utf编码格式区间
re_han = re.compile("([^a-zA-Z0-9\u4E00-\u9FD5\r\n\s]+)", re.U)


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
    f=open(infiles,encoding="utf-8")
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
    corpus("../Dataset/segment/training/pku_training.txt","../Dataset/segment/training/pku_corpus.txt")



