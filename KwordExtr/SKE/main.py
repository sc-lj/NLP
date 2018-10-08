#encoding=utf-8


try:
    import textPreprocessing,semanticsCount,statisticsCount
except:
    from . import textPreprocessing, semanticsCount, statisticsCount

from breadability.readable import Article
import math

def extract_html(content):
    article = Article(content)
    annotated_text=article.main_text
    paragraphs=""
    for paragraph in annotated_text:
        sentences=""
        for text, annotations in paragraph:
            sentences+=text
        paragraphs+=sentences
    return paragraphs


def SemankeyWord(content, title,skenum=None):
    content=extract_html(content)
    # 逻辑结构
    # 1、文本预处理(分词与词性标注、词语过滤、词语相关信息记录)
    wordsStatisticsData, wordsData = textPreprocessing.word_segmentation(content, title)
    # 2、词语语义贡献值计算(计算词语语义相似度、构建词语语义相思网络、计算词语居间度密度)
    intermediaryDensity = semanticsCount.intermediaryDegreeDensity(content, title)
    # 3、计算词语统计特征值
    # keywordDatas = statisticsCount.tfidf()
    wordsStatisticsData = statisticsCount.wordsStatistics(wordsStatisticsData)
    # 4、计算词语关键度
    # 算法基础设定
    # 语义贡献值权重
    vdw = 0.6
    # 统计特征值权重
    tw = 0.4
    # 统计特征位置上权重
    locw1, locw2, locw3 = 0.5, 0.3, 0.3
    # 统计特征词长权重
    lenw = 0.01
    # 统计特征值中词性权重
    posw = 0.5
    # 统计特征中TF-IDF权重
    tfidfw = 0.8

    # 对收集到的词语进行重新遍历
    ske = {}
    for key in wordsStatisticsData.keys():
        # 取语义贡献值(假如居间度密度集合中不存在,补充为0)
        if intermediaryDensity.get(key):
            vdi = intermediaryDensity.get(key)
        else:
            vdi = 0

        # 暂时未加tfidf权值
        score = vdw * vdi + tw * (locw1 * float(wordsStatisticsData[key][0]) + lenw * int(len(key)) + posw * float(
                wordsStatisticsData[key][1]))
        ske[key] = score

    ske = sorted(ske.items(), key=lambda d: d[1], reverse=True)  # 降序排列
    skelen=len(ske)
    if skenum is None:
        ske=ske[:math.ceil(skelen/3)]
    else:
        ske=ske[:skenum]
    words=[word for word,_ in ske]
    words=",".join(words)
    return words

if __name__ == "__main__":
    # 进行关键词提取的文章
    content="""<video id='video' src='http://vod-xhpfm.oss-cn-hangzhou.aliyuncs.com/NewsVideo/201809/20180914121600_7547.mp4' poster='http://img-xhpfm.oss-cn-hangzhou.aliyuncs.com/News/201809/20180914121606_1611.jpg'width='100%' controls='controls' ></video><img src="https://dot.xinhuazhiyun.com/logserver/1.gif?logtype=rss&amp;version=1.0.0&amp;id=4550093&amp;source=7fb40b54bd6a4550" hidden/><p style="">原标题:葫芦河之变</p>【简介】浇上好水才能种出好菜。断流３０多年的葫芦河在宁夏固原市西吉县的治理下通水了。一河清水将给两岸１４．２万人脱贫致富带来新希望。　<link href="https://dot.xinhuazhiyun.com/logserver/1.gif?logtype=rss&amp;version=1.0.0&amp;id=4550093&amp;source=7fb40b54bd6a4550" rel="stylesheet" type="text/css" /> """

    title="葫芦河之变"
    print(SemankeyWord(content, title))

