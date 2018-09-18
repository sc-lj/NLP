# coding:utf-8
"""Data batchers for data described in ..//data_prep/README.md."""

import glob,jieba
import random
import struct
import sys,re
from database import MySQL
from tensorflow.core.example import example_pb2
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
  """Vocabulary class for mapping words and ids."""

  def __init__(self, vocab_file, max_size):
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0

    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.split()
        if len(pieces) != 2:
          sys.stderr.write('Bad line: %s\n' % line)
          continue
        if pieces[0] in self._word_to_id:
          raise ValueError('Duplicated word: %s.' % pieces[0])
        self._word_to_id[pieces[0]] = self._count
        self._id_to_word[self._count] = pieces[0]
        self._count += 1
        if self._count > max_size:
          raise ValueError('Too many words: >%d.' % max_size)

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


def ExampleGen1(data_path, num_epochs=None):
  """Generates tf.Examples from path of data files.

    Binary data format: <length><blob>. <length> represents the byte size
    of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
    the tokenized article text and summary.

  Args:
    data_path: path to tf.Example data files.
    num_epochs: Number of times to go through the data. None means infinite.

  Yields:
    Deserialized tf.Example.

  If there are multiple files specified, they accessed in a random order.
  """
  epoch = 0
  while True:
    if num_epochs is not None and epoch >= num_epochs:
      break
    filelist = glob.glob(data_path)
    assert filelist, 'Empty filelist.'
    random.shuffle(filelist)
    for f in filelist:
      reader = open(f, 'rb')
      while True:
        len_bytes = reader.read(8)
        if not len_bytes: break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        yield example_pb2.Example.FromString(example_str)

    epoch += 1

def check_html(content):
    rec=re.compile("<.*>")
    return rec.match(content)

def extract_html(content):
    article = Article(content)
    annotated_text=article.main_text
    paragraphs=""
    split_sent=['。','？','！']

    #  将双分号里面的句子不进行分割。
    Dquotes=['"','“','”']
    for paragraph in annotated_text:
        print(paragraph)
        sentences=""
        for text, annotations in paragraph:
            sentences+=text
        sentences=list(jieba.cut(sentences))
        quote=False
        newsentences=""
        newsentences+=" "+PARAGRAPH_START+" "+SENTENCE_START+" "
        for word in sentences:
            if word in Dquotes and not quote:
                quote=True
                newsentences+=word+" "
            elif word in Dquotes and quote:
                quote=False
                newsentences+=word+" "
            elif quote:
                newsentences+=word+" "
            elif word in split_sent and not quote:
                newsentences+=word+" "
                newsentences+=SENTENCE_END+" "
                newsentences+=SENTENCE_START+" "
            else:
                newsentences+=word+" "
        newsentences=newsentences[:-len(SENTENCE_START+" ")]
        newsentences+=PARAGRAPH_END+" "
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
        brief=extract_html(brief)
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


def GetExFeatureText(ex, key):
  return ex.features.feature[key].bytes_list.value[0]


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

    content=u"图为第三季港澳台学子金华活动闭幕式合影　　吴俊鹏摄来自港澳台地区的37名高校学子近日来到黄大仙故里浙江金华，参加第三季“台港澳学子走进‘侨仙’故里金华”活动，开启了为期10天的文化寻根之旅。回家的感觉真好“关上房门是星级酒店、打开房门是温馨家庭、走出家门是乡野生活”，这是金华首创的“家+”模式。在前两季“台港澳学子走进金华古村落”活动中，它深受学子欢迎。学子们入住“家+”有一定的仪式。“家+”的住家们手拿对联站在村口，学子们则需要在另一边的对联中，找出与自己入住之处相呼应的对联。“少小离家老大回，下一句是？”“乡音无改鬓毛衰。”“哈哈，孩子，欢迎你到我家。”配对成功后，学生即可跟着住家回去。来自澳门、目前在中国人民大学读研的吴海琴对此感触特别深：“这里的一山一水、一家一户、每个人一下子变得非常亲切，好像前世注定的亲人一样，注定在某个地方相遇，真的太美！”每当夜幕降临，炊烟升起，在古村落的一幢幢房子里，一户户“家+”灯火通明，港澳台学子与住家叔叔阿姨们一起用餐，一道聊天，一片欢声笑语。金华市港澳办主任章宏说，金华要通过“家+住宿”“家+餐饮”“家+活动”，特别是“家+特殊人文”，打造环境古朴、设施现代、人文诚善，既有住家体验又有旅行情怀，还有研学修行的台港澳学子的“情感驿站”和“心灵港湾”。文化传承的使者作为中华优秀传统文化的一种精华浓缩，黄大仙文化已经在22个国家和地区传播，金华传统村落文化也留在了台港澳与海外高校学子的记忆里。“我们两岸三地青年一代有责任当好保护和传承中华传统文化的使者。”来自香港大学的丁泽棋认为，金华要与粤港澳联手，更好地保护和发扬“黄大仙文化”，使其成为东西方文化兼容的典型代表。澳门城市大学万淑娟同学则建议，金华要用好作为黄大仙文化发源地、港澳影视演员常年在横店拍摄作品的独特优势，精心制作黄大仙文化和传统村落文化影视作品及动漫产品，延伸文化产业链，使自身成为与“一带一路”沿线国家对接、凝聚海内外华人的重要纽带。活动组织者之一、北大金华选调生吴俊鹏说，每一季港澳台学子进古村活动中，都会开展以传统文化的传承、融合、发展为主线的海峡两岸暨港澳青年论坛，以此推进两岸暨港澳青年主动融入国家发展大局，共绘中国梦。乡梓故里的依恋“黄大仙在港澳地区可谓家喻户晓，寄托了我们祖祖辈辈对中华传统文化的归属与依恋。没想到黄大仙来自金华，而且有1600多年历史了。”第一次来金华文化寻根的香港大学学生蓝晓珺感慨不已。金华是黄大仙的祖庭所在地。晋代黄大仙因擅长炼丹和医术，一生劝善扬善、惩恶除奸、有求必应，泽被一方，民众遂为之修祠，世代祀奉。随着金华乡亲旅居港澳地区和海外，黄大仙信仰也走向港澳和海外，并成为众多华侨华人心中的“侨仙”。金华一直是港澳台地区和海外信众的朝圣地，两岸三地经常在这里联合举办庙会和旅游节。"
    extract_html(content)

