# coding:utf-8


import os,sys
syspath=os.path.abspath(__file__)

while True:
    sys.path.append(syspath)
    try:
        from script.SQLConfig import *
        break
    except:
        syspath=os.path.dirname(syspath)

from sklearn.feature_extraction.text import TfidfVectorizer
import re,json,jieba
from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict
from breadability.readable import Article
from Utils import *


def stopwords():
    """
    读取停用词
    :return:
    """
    with open('../script/stopwords.txt','r') as f:
        stopword=f.read()
    return stopword

def cutWords(content):
    """
    分句子后，然后分词,并去掉停用词
    :param content:
    :return:
    """
    content=content.lower()
    stopword = stopwords()
    splitdot=re.compile('[。，？！：,.、;-]')# 断句的标点呼号
    sentences=splitdot.split(content)# 切分成多个语句
    words=[]
    for sentence in sentences:
        rule = re.compile(r"[^\u4e00-\u9fa5]")  # 只留中文。
        sentence=rule.sub('',sentence)
        word=jieba.cut(sentence)#结巴对句子进行分词
        words.extend(list(word))

    # 过滤停用词
    filterstopword=list(filter(lambda x:x if x not in stopword else None, words))
    return filterstopword

def vocabcorpus(files,obj=None):
    """
    写入或读取字典
    :param files: 文件名
    :param obj: 要写入的数据
    :return:
    """
    if not obj:
        f=open(files,"r")
        data=f.read()
        vocab=json.loads(data)
        f.close()
        return vocab
    elif obj:
        f=open(files,"w")
        data=json.dumps(obj)
        f.write(data)
        f.close()


def tfIdf(textIds):
    """
    tfidf算法对于短文本，在这就是标题，是不适用的。因为单词在标题中的tf近似为1，即相当于只计算了idf。可以用来处理新闻内容。
    :param textIds: 传入的数据是{id:text}
    :return:文本的tfidf和文本的短词
    """
    if not isinstance(textIds,dict):
        raise("必须传入字典形式的文本，key为文本的id，value是文本的内容")

    corpus=[]
    IDs=[]
    for ids,content in textIds.items():
        IDs.append(ids)
        corpus.append(" ".join(cutWords(content)))
    # 自己传入字典
    # vocab=vocabcorpus("./script/vocab.json")
    vectorizer=TfidfVectorizer(vocabulary=None)
    tfidf=vectorizer.fit_transform(corpus)
    # 得到所有词
    wordscorpus=vectorizer.get_feature_names()
    # 文本库的tfidf矩阵
    textsarray=tfidf.toarray()

    textsTfidf={}# 文本{textid:}
    textslist={}
    for i in range(len(IDs)):
        texttfidf={}# 存储单个文本的tfidf值,{word:tfidf}
        textlist=[]
        for j in range(len(wordscorpus)):
            if textsarray[i][j]>0:
                texttfidf[wordscorpus[j]]=textsarray[i][j]
                textlist.append(wordscorpus[j])
        textsTfidf[IDs[i]]=texttfidf
        textslist[IDs[i]]=textlist
    return textsTfidf,textslist

def genCorpus():
    """
    从新闻数据库中读取文本生成语料库
    :return:
    """
    titlecorpus=defaultdict(int)
    contentcorpus=defaultdict(int)
    db=login_sql()
    cursor=db.cursor()
    for dbase in ["crawldata","main"]:
        sql="select title,content from %s where 1"%dbase
        cursor.execute(sql)
        for title,content in cursor.fetchall():
            if len(content)<=10:
                continue
            content=processHtml(content)
            contents=cutWords(content)
            for con in contents:
                contentcorpus[con]+=1

            words=cutWords(title)
            for word in words:
                titlecorpus[word]+=1

    corpus=json.dumps(titlecorpus)
    with open('../script/titlecorpus.json','w') as f:
        f.write(corpus)
    with open('../script/contentcorpus.json','w') as f:
        f.write(corpus)

def processHtml(html):
    """
    处理html语言
    :param html:
    :return:
    """
    _article = Article(html)
    annotated_text = _article.main_text
    sentences = []
    for paragraph in annotated_text:
        current_text = ""
        for text, annotations in paragraph:
            current_text += " " + text
        sentences.append(current_text)
    return "".join(sentences)


def mutualInfo(textIds):
    """
    通过互信息提取文本的关键字
    :param textIds:
    :return:
    """






if __name__ == '__main__':
    # a=cutWords("文本挖掘是一门交叉性学科,涉及数据挖掘、机器学习、模式识别、人工智能、统计学、计算机语言学、计算机网络技术、信息学等多个领域。文本挖掘就是从大量的文档中发现隐含知识和模式的一种方法和工具,它从数据挖掘发展而来,但与传统的数据挖掘又有许多不同。文本挖掘的对象是海量、异构、分布的文档(web);文档内容是人类所使用的自然语言,缺乏计算机可理解的语义。传统数据挖掘所处理的数据是结构化的,而文档(web)都是半结构或无结构的。所以,文本挖掘面临的首要问题是如何在计算机中合理地表示文本,使之既要包含足够的信息以反映文本的特征,又不至于过于复杂使学习算法无法处理。在浩如烟海的网络信息中,80%的信息是以文本的形式存放的，WEB文本挖掘是WEB内容挖掘的一种重要形式。")
    # tfIdf({"1":'机器学习是一门多领域交叉学科',"2":'涉及概率论、统计学、逼近论、凸分析、算法复杂度理论等多门学科'})
    genCorpus()
    html="""
    	<p>1月26日下午2点，中共成都市委党校经济学教研部副主任、副教授，西南财经大学、四川大学、西南交通大学、电子科技大学四所髙校客座教授童晶，受邀走进成都市武侯区玉林街道“天府讲堂”，就如何学习贯彻党的十九大精神，有效开展基层建设，从自身经验出发，为辖内居民做了精彩详尽的讲演。&nbsp;</p><p><strong>迈入新时代 踏上新征程</strong></p><p><strong>党的十九大从“新”开始</strong></p><p><img src="http://images.cdsb.com/Uploads/micropub_posts/image/20180126/1516951352136260.png?imageView2/0/w/1080/q/60|watermark/1/image/aHR0cDovL3N0YXRpYy5jZHNiLmNvbS9taWNyb3B1Yi9zdGF0aWMvd2F0ZXJtYXJrLnBuZw==/ws/0.2" title="1516951352136260.png" alt="66.png"/></p><p>“党的十九大开幕式上，习总书记铿锵有力的话语，传遍神州大地，直抵人心。统计发现，3万多字的十九大报告中，‘新’字出现了174次，其中‘新时代’一词便出现36次。”</p><p>童晶表示，“经过全国人民的不懈努力，过去五年，我国在政治、经济、科技、文化、环境等各方面纷纷取得举世瞩目的成绩，如今，新时代的大门已经开启。”</p><p>“新时代都有哪些变化？”童晶告诉记者，“最显著地就是社会主要矛盾的转变，这是关系全局的历史性变化。”她说，“随着社会进步发展，人民美好生活需要日益广泛，不仅对物质文化生活提出了更高要求，而且在民主、法治、公平、正义、安全、环境等方面的要求日益增长。我们的社会主要矛盾，已由‘人民日益增长的物质文化需要同落后的社会生产之间的矛盾’，转变为‘人民日益增长的美好生活需要和不平衡不充分的发展之间的矛盾’。”</p><p>“但我们必须认识到，我国社会主要矛盾的变化，没有改变我们对我国社会主义所处历史阶段的判断,我国仍处于并将长期处于社会主义初级阶段的基不国情没有变，我国是世界最大发展中国家的国际地位没有变。”</p><p>童晶指出，“为决胜全面建成小康社会，夺取新时代中国特色社会主义伟大胜利，实现中华民族伟大复兴的中国梦，我们应该坚持党对一切工作的领导，坚持以人民为中心，坚持全面深化改革，坚持新发展理念，坚持人民当家做主，坚持全面依法治国，坚持社会主义核心价值体系，坚持在发展中保障和改善民生，坚持人与自然和谐共生，坚持总体国家安全观，坚持党对人民军队的绝对领导，坚持一国两制推进祖国统一，坚持推动构建人类命运共同体，坚持全面从严治党。”</p><p>分享会末尾，童晶说道，“作为共产党员，大家应当不忘初心，牢记使命，高举中国特色社会主义伟大旗帜，谨记自身使命与任务，担起肩上的责任，抓住当前和今后一个时期基层建设的关键，拿出实实在在的举措，把十九大确定的近期、中期、长期目标任务有计划、有秩序地加以推进，在全面建成小康社会、迈向现代化的进程中走在前列。”&nbsp;</p><p><strong>贯彻落实党的十九大精神</strong></p><p><strong>让基层党建工作越来越坚实</strong></p><p><img src="http://images.cdsb.com/Uploads/micropub_posts/image/20180126/1516951389385759.png?imageView2/0/w/1080/q/60|watermark/1/image/aHR0cDovL3N0YXRpYy5jZHNiLmNvbS9taWNyb3B1Yi9zdGF0aWMvd2F0ZXJtYXJrLnBuZw==/ws/0.2" title="1516951389385759.png" alt="6633.png"/></p><p>“童主任的讲说高屋建瓴，立意深远，内涵丰富，思想深刻，情感真挚地为大家讲述了新时代的变化，让我们更直观地领会理解到党的十九大精神，对今后的生活更加充满期待和向往，对今后基层工作的开展更加充满信心。”谈及参会感受，玉林街道党员同志张恒直言受益匪浅。<br/></p><p>“党的十九大报告指出，党的基层组织是确保党的路线方针政策和决策部署贯彻落实的基础，要以提升组织力为重点，突出政治功能，把企业、农村、机关、学校、科研院所、街道社区、社会组织等基层党组织建设成为宣传党的主张、贯彻党的决定、领导基层治理、团结动员群众、推动改革发展的坚强战斗堡垒。”</p><p>张恒告诉记者，“作为基层党员，我们要贯彻落实十九大精神，永远把人民对美好生活的向往作为奋斗目标；要进一步坚定‘扎根基层、融入基层、奉献基层’、‘全心全意为人民服务’的理想信念；要做勤于学习的表率，做崇尚实干的表率，继续学深学透党的基本理论，把学习贯彻好党的十九大精神作为首要任务。”</p><p>张恒表示，“玉林街道全体工作人员，都将牢记习近平总书记的嘱托，担负好教育党员、管理党员、监督党员和组织群众、宣传群众、凝聚群众、服务群众的职责，都将扎根基层，服务百姓，立足本职岗位，尽忠履职，将十九大精神带到社区，落到实处，把工作做到辖区居民心坎里，让基层党建工作越来越坚实，把社区打造成为服务居民的平台、密切党群干群关系的桥梁和纽带，让老百姓越来越拥护党组织，生活更有奔头，真正让幸福在家门口实现，不断增强居民的幸福感、安全感和获得感。”&nbsp;</p><p>成都商报客户端记者 沈兴超</p>
    """
    # processHtml(title)
