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
        self._word_count = 0
        self._topic_to_id={}
        self._topic_count=0
        self._word_to_topic=[]# 词汇表和主题词汇表的映射关系
        self.indicator=[]

        tvocab_f=open(tvocab_file,'r')
        t_lines=tvocab_f.readlines()
        t_lines=[line.strip() for line in t_lines]
        for t_line in t_lines:
            self._topic_to_id[t_line]=self._topic_count
            self._topic_count+=1

        with open(vocab_file, 'r',encoding='utf-8') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    sys.stderr.write('Bad line: %s\n' % line)
                    continue
                if pieces[0] in self._word_to_id:
                    raise ValueError('Duplicated word: %s.' % pieces[0])
                if pieces[0] in t_lines:
                    self._word_to_topic.append([self._word_count,self._topic_to_id[pieces[0]]])
                    self.indicator.append(1)
                else:
                    self.indicator.append(0)

                self._word_to_id[pieces[0]] = self._word_count
                self._id_to_word[self._word_count] = pieces[0]
                self._word_count += 1

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
        return self._word_count

    def WordToTopic(self):
        return self._word_to_topic

    def TopicIds(self):
        return self._topic_count


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
    mysql=MySQL(sqldb="HWCom")
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
            yield (title,content,brief)

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
    items=ExampleGen(1)
    import json
    f=open("chidata.txt",'a')
    a=1
    for title,content, brief in items:
        data1={"title":title,"content":content,"brief":brief}
        data=json.dumps(data1)
        f.write(data+"\n")
        a+=1
    print(a)
    f.close()
