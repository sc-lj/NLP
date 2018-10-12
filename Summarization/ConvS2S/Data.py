# coding:utf-8
"""Data batchers for data described in ..//data_prep/README.md."""

import glob,jieba
import sys,re
from database import MySQL
from breadability.readable import Article

# Special tokens
PARAGRAPH_START = '<p>'
PARAGRAPH_END = '</p>'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
DOCUMENT_START = '<d>'
DOCUMENT_END = '</d>'


class Vocab(object):
    def __init__(self, vocab_file,tvocab_file):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0
        self._id_to_topic={}

        tvocab_f=open(tvocab_file,'r')
        t_lines=tvocab_f.readlines()
        t_lines=[line.strip() for line in t_lines]
        with open(vocab_file, 'r',encoding='utf-8') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    sys.stderr.write('Bad line: %s\n' % line)
                    continue
                if pieces[0] in self._word_to_id:
                    raise ValueError('Duplicated word: %s.' % pieces[0])
                if pieces[0] in t_lines:
                    self._id_to_topic[self._count]=1
                else:
                    self._id_to_topic[self._count]=0
                self._word_to_id[pieces[0]] = self._count
                self._id_to_word[self._count] = pieces[0]
                self._count += 1

    def CheckVocab(self, word):
        if word not in self._word_to_id:
            return None
        return self._word_to_id[word]
  
    def WordToId(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def IdToWord(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('id not found in vocab: %d.' % word_id)
        return self._id_to_word[word_id]

    def NumIds(self):
        return self._count


def check_html(content):
    rec=re.compile("<.*>")
    return rec.match(content)


def extract_html(content,is_content=True):
    if is_content:
        article = Article(content)
        annotated_text=article.main_text
    else:
        annotated_text=[((content,None),)]
    paragraphs=""
    split_sent=['。','？','！','!','?']

    #  将双分号里面的句子不进行分割。
    Dquotes=['"','“','”']
    for paragraph in annotated_text:
        sentences=""
        for text, annotations in paragraph:
            sentences+=text
        sentences = re.sub("(（.*?）)?(\(.+\))?(编译.+)?(责编：.+)?", "", sentences)
        sentences = re.sub("(本文系版权作品，未经授权严禁转载。.*)\s?(责编)?", "", sentences)
        sentences = re.sub("\d", "#", sentences)
        sentences=" ".join(jieba.cut(sentences))
        if len(sentences)==0:
            continue
        quote=False
        newsentences=""
        newsentences+=" "+PARAGRAPH_START+" "+SENTENCE_START+" "
        for word in sentences:
            if word in Dquotes and not quote:
                quote=True
                newsentences+=word
            elif word in Dquotes and quote:
                quote=False
                newsentences+=word
            elif quote:
                newsentences+=word
            elif word in split_sent and not quote:
                newsentences+=word
                newsentences+=" "+SENTENCE_END+" "
                newsentences+=SENTENCE_START+" "
            else:
                newsentences+=word
        if len(newsentences)-newsentences.rfind(SENTENCE_START+" ")==4:
            newsentences=newsentences[:-len(SENTENCE_START+" ")]
        else:
            newsentences+=" "+SENTENCE_END
        newsentences+=" "+PARAGRAPH_END
        paragraphs+=newsentences
    return paragraphs


def ExampleGen(num_epochs=None):
    epoch = 0
    mysql=MySQL()
    mysql.login()
    cursor=mysql.get_cursor()
    while True:
        if num_epochs is not None and epoch >= num_epochs:
            break

        sent="select title,brief,content from news where brief !=''"
        cursor.execute(sent)
        for rows in cursor.fetchall():
            title, brief, content=rows
            content=extract_html(content)
            brief=extract_html(brief,False)
            yield (content,brief)

        epoch += 1


def Pad(ids, pad_id, length):
    """Pad or trim list to len length.

    Args:
    ids: list of ints to pad
    pad_id: what to pad with
    length: length to pad or trim to

    Returns:
    ids trimmed or padded with pad_id
    """
    assert pad_id is not None
    assert length is not None

    if len(ids) < length:
        a = [pad_id] * (length - len(ids))
        return ids + a
    else:
        return ids[:length]


def GetWordIds(text, vocab, pad_len=None, pad_id=None):
    """Get ids corresponding to words in text.

    Assumes tokens separated by space.

    Args:
    text: a string
    vocab: TextVocabularyFile object
    pad_len: int, length to pad to
    pad_id: int, word id for pad symbol

    Returns:
    A list of ints representing word ids.
    """
    ids = []
    for w in text.split():
        i = vocab.WordToId(w)
        if i >= 0:
            ids.append(i)
        else:
            ids.append(vocab.WordToId(UNKNOWN_TOKEN))
    if pad_len is not None:
        return Pad(ids, pad_id, pad_len)
    return ids


def Ids2Words(ids_list, vocab):
    """Get words from ids.

    Args:
    ids_list: list of int32
    vocab: TextVocabulary object

    Returns:
    List of words corresponding to ids.
    """
    assert isinstance(ids_list, list), '%s  is not a list' % ids_list
    return [vocab.IdToWord(i) for i in ids_list]

def SnippetGen(text, start_tok, end_tok, inclusive=True):
    """Generates consecutive snippets between start and end tokens.

    Args:
    text: a string
    start_tok: a string denoting the start of snippets
    end_tok: a string denoting the end of snippets
    inclusive: Whether include the tokens in the returned snippets.

    Yields:
    String snippets
    """
    cur = 0
    while True:
        try:
            start_p = text.index(start_tok, cur)
            end_p = text.index(end_tok, start_p + 1)
            cur = end_p + len(end_tok)
            if inclusive:
                yield text[start_p:cur]
            else:
                yield text[start_p+len(start_tok):end_p]
        except ValueError as e:
            raise StopIteration('no more snippets in text: %s' % e)


def ToSentences(paragraph, include_token=True):
    """Takes tokens of a paragraph and returns list of sentences.

    Args:
    paragraph: string, text of paragraph
    include_token: Whether include the sentence separation tokens result.

    Returns:
    List of sentence strings.
    """
    s_gen = SnippetGen(paragraph, SENTENCE_START, SENTENCE_END, include_token)
    return [s for s in s_gen]


if __name__ == '__main__':
    content="福建省加快推进海洋强省重大项目建设"
    content1="""<div class="contentMain">
                <p style="text-indent: 2em;"><strong><a href="http://www.haiwainet.cn/" target="_blank" class="keylink">海外网</a>9月19日电</strong> 波兰领导人周二（18日）敦促美国总统特朗普考虑在波兰建立一个永久性的美国军事基地，以防范俄罗斯。特朗普回应称，非常认真考虑应波兰要求，在波兰永久派驻美军。</p><p style="text-align:center"><img src="http://images.haiwainet.cn/20180919/1537303125961392.jpg" title="1537303125961392.jpg" alt="8(1).jpg"></p><p style="text-indent: 2em; text-align: center;">特朗普（右）在椭圆形办公室见杜达。（图：美联社）</p><p style="text-indent: 2em;">据《美联社》（AP）消息，波兰总统杜达（Andrzej Duda）在白宫与特朗普的一对一会谈中发出邀请，称该国面临俄罗斯在该地区增加的军事活动之压力，波兰希望将在波兰部署的3000名美军升级为更大的永久存在。杜达甚至提供诱人的激励：新建的永久性基地将被命名为“特朗普堡”（Fort Trump）。</p><p style="text-indent: 2em;">特朗普在与杜达的联合<a href="http://world.haiwainet.cn/" target="_blank" class="keylink">新闻</a>发布会上表示，他正在考虑这个提议，并指出波兰已向美国提供超过20亿美元以支付这笔费用。就他而言，杜达相信特朗普已朝着在波建立永久性美军基地迈出了重要的一步。特朗普表示，两国领导人同意加强国防、能源和商业关系，并表示两国之间的联盟“从未变得比现在更强大”。</p><p style="text-indent: 2em;">据悉，美军在去年起，向波兰及周边国家派遣为数3000人的装甲旅和重装部队；今年年中有波兰媒体报道指，波兰希望美军一个装甲师永久部署当地，愿意出资最多20亿美元协助建造相关军事设施。俄罗斯克里姆林宫则警告，美方行动威胁了俄罗斯家门口的安全。（编译/海外网 侯兴川）</p><p style="text-indent: 2em;">本文系版权作品，未经授权严禁转载。海外视野，中国立场，登陆人民日报海外版官网——海外网www.haiwainet.cn或“海客”客户端，领先一步获取权威<a href="http://news.haiwainet.cn/" target="_blank" class="keylink">资讯</a>。</p>                <p class="writer">责编：侯兴川 </p>
                <div style="display:none;">
                                                 <img src="http://images.haiwainet.cn/2018/0919/20180919044544869.jpg" alt="">
                                       </div>
                <div class="page"></div>
            </div>"""
    tosentence=extract_html(content,False)
    a=ToSentences(tosentence)
