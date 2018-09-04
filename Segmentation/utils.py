# coding:utf-8

import codecs,re


re_han_default = re.compile("([\u4E00-\u9FD5]+)", re.U)
re_skip_default = re.compile("(\r\n|\s)", re.U)
# 中文utf编码格式区间
re_han = re.compile("([^a-zA-Z0-9\u4E00-\u9FD5\r\n\s]+)", re.U)

# 用该标示符来标记在中文中具有天然的分割符号，如""、《》。等，并将其分割标示为O
PAD="PAD"

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


 # 5-tags for character tagging:B(Begin),E(End),M(Middle),S(Single)，O(专门标示中文具有天然的分割符号)
def corpus(infiles,outfile):
    """
    将训练数据集转换成对应模型可训练的形式
    :param infiles:
    :param outfile:
    :return:
    """
    f=open(infiles,'r',encoding="utf-8")
    outdata = codecs.open(outfile, 'w', 'utf-8')
    lines=f.readlines()
    for line in lines:
        line=strQ2B(line)
        pad=0
        sentence=re_han.sub(PAD,line)
        for words in re.split(" |\u0020",sentence):
            if len(words.strip())==0:
                continue
            if len(words)==1 or words=="PAD":
                # 这是预防连续出现两个PAD字符
                if words==PAD and pad==0:
                    outdata.write(words+"\tO\n")
                    pad+=1
                elif words!=PAD:
                    outdata.write(words+"\tS\n")
            else:
                pad=0
                words=re_han_default.split(words)
                for word in words:
                    if len(word.strip()) == 0:
                        continue
                    outdata.write(word[0]+"\tB\n")
                    for i in range(1,len(word)-1):
                        outdata.write(word[i] + "\tM\n")
                    outdata.write(word[-1]+"\tE\n")
        outdata.write("\n")

    f.close()
    outdata.close()


if __name__ == '__main__':
    corpus("../Dataset/segment/training/pku_training.txt","./pku_corpus.txt")


