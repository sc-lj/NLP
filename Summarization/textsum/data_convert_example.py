#coding:utf-8
"""Example of Converting TextSum model data.
Usage:
python data_convert_example.py --command binary_to_text --in_file data/data --out_file data/text_data
python data_convert_example.py --command text_to_binary --in_file data/text_data --out_file data/binary_data
python data_convert_example.py --command binary_to_text --in_file data/binary_data --out_file data/text_data2
diff data/text_data2 data/text_data
"""

import struct
import sys,re,jieba
from collections import defaultdict
import pandas as pd
import tensorflow as tf
from tensorflow.core.example import example_pb2
from database import MySQL
from breadability.readable import Article
import data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('command', 'binary_to_text',
                           'Either binary_to_text or text_to_binary.'
                           'Specify FLAGS.in_file accordingly.')
tf.app.flags.DEFINE_string('in_file', 'data/data', 'path to file')
tf.app.flags.DEFINE_string('out_file', 'data/text_data', 'path to file')

def _binary_to_text():
  reader = open(FLAGS.in_file, 'rb')
  writer = open(FLAGS.out_file, 'w')
  while True:
    len_bytes = reader.read(8)
    if not len_bytes:
      sys.stderr.write('Done reading\n')
      return
    str_len = struct.unpack('q', len_bytes)[0]
    tf_example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
    tf_example = example_pb2.Example.FromString(tf_example_str)
    examples = []
    for key in tf_example.features.feature:
      examples.append('%s=%s' % (key, tf_example.features.feature[key].bytes_list.value[0]))
    writer.write('%s\n' % '\t'.join(examples))
  reader.close()
  writer.close()


def _text_to_binary():
  inputs = open(FLAGS.in_file, 'r').readlines()
  writer = open(FLAGS.out_file, 'wb')
  for inp in inputs:
    tf_example = example_pb2.Example()
    for feature in inp.strip().split('\t'):
      (k, v) = feature.split('=')
      tf_example.features.feature[k].bytes_list.value.extend([v])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))
  writer.close()


def main(unused_argv):
  assert FLAGS.command and FLAGS.in_file and FLAGS.out_file
  if FLAGS.command == 'binary_to_text':
    _binary_to_text()
  elif FLAGS.command == 'text_to_binary':
    _text_to_binary()


def genVocab(vocabfile):
    mysql=MySQL()
    mysql.login()
    cursor=mysql.get_cursor()
    sent="select title,brief,content from news where 1"
    cursor.execute(sent)

    vocab=defaultdict(int)
    def imdict(ab):
        ab=jieba.cut(ab)
        for a in ab:
            if len(a)==0:
                continue
            vocab[a]+=1

    da=[]
    for sent in cursor.fetchall():
        title, brief, content=sent
        title=re.sub("\d","#",title)
        imdict(title)

        brief= re.sub("摘要：","",brief)
        brief= re.sub("\d","#",brief)
        imdict(brief)
        brieflen=len(brief)

        content=re.sub("资料图（图源：.*?）","",content)
        content=re.sub("(本文系版权作品，未经授权严禁转载。.*)\s?责编","责编",content)
        content=re.sub("(.编译.*责编：.+)","",content)
        content=re.sub("\d","#",content)
        imdict(content)
        contentlen=len(content)
        da.append([brieflen,contentlen])
    data = pd.DataFrame(columns=["brief", "content"],data=da)
    data=data[data['brief']>0]
    mysql.close()
    print(len(vocab))
    vocab={key:value for key,value in vocab.items() if value>=5}
    print(len(vocab))
    print(sum(vocab.values()))
    with open(vocabfile,'w') as f:
        for word,num in vocab.items():
            f.write(word+" "+str(num)+"\n")
    data.to_csv("./data/len.csv",index=False)

def extract_html(content):
    article = Article(content)
    annotated_text=article.main_text
    paragraphs=""
    split_sent=['。','？','！']

    #  将双分号里面的句子不进行分割。
    Dquotes=['"','“','”']
    for paragraph in annotated_text:
        sentences=""
        for text, annotations in paragraph:
            sentences+=text
        quote=False
        newsentences=""
        newsentences+=" "+data.PARAGRAPH_START+" "+data.SENTENCE_START+" "
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
                newsentences+=" "+data.SENTENCE_END
                newsentences+=" "+data.SENTENCE_START+" "
            else:
                newsentences+=word
        newsentences=newsentences[:-len(" "+data.SENTENCE_START+" ")]
        newsentences+=" "+data.PARAGRAPH_END+" "
        paragraphs+=newsentences

    return paragraphs

def toSentence():
    mysql=MySQL()
    mysql.login()
    cursor=mysql.get_cursor()
    sent="select title,brief,content from news where 1"
    cursor.execute(sent)
    for content in cursor.fetchall():
        content=extract_html(content)





if __name__ == '__main__':
    # tf.app.run()
    # genVocab("./data/chivocab")
    content="""<div class="left_zw" >  <p>　　<a target="_blank" href="http://www.chinanews.com/">中新网</a>7月13日电 据澳洲网报道，最近，澳大利亚墨尔本Elwood地区的多处地方都莫名出现了很多奶酪，这些奶酪有的散落在院子里，有的散落在草丛里，有些奶酪还已经发霉长毛。当地议会近日警告民众，要远离这些神秘的奶酪，有可能这些奶酪是被当做诱饵投放在多处。</p><table border="0" cellspacing="0" cellpadding="0" align="left" ><tr><td></td></tr></table><p>　　<strong>民众应防止宠物狗误食</strong></p><p>　　报道称，由于墨尔本Elwood地区多处出现不明来源的大块奶酪，已经引起当地民众的不安和恐慌，近日Port Phillip议会表示，他们已经将两块奶酪送去接受检测，并且强烈要求民众不要让自家的宠物狗误食这些奶制品。该案件的调查负责人史密斯(Peter Smith)表示，</p><p>　　人们应该将这些奶酪小心的丢弃在垃圾桶里，之后再好好清洁一下双手。“虽然目前没有迹象表明这些奶酪被人下毒，但是我们应该谨慎对待，这些奶酪正在被送往检测机构进行检测，以确保它们不是用来引诱犬类上钩的诱饵，”史密斯说，“就算这些奶酪经检测没有问题，我们也建议狗的主人看紧宠物狗，不要让它们吃得太多，因为过量的奶制品会引发狗狗呕吐。”</p><p>　　报道称，当地议会会持续关注该案件的发展，查清是谁在随意丢弃这些奶制品。据当地居民反映，这些奶制品在几周之前就出现了，有人在树丛中、草地上、公路旁和居民区内，甚至是通往沙滩的Tiuna Grove大街、Byrne大街和Bendigo大街上都看到过这些奶酪，人们都将此事称为“Elwood神秘奶酪案件”。</p><p>　　<strong>神秘奶酪来源扑朔迷离</strong></p><p>　　据悉，当地很多民众认为是狐狸将这些奶酪散落的到处都是，但是目前还没有人知道这些奶酪的确切来历。</p><p>　　有人称是附近一位老奶奶丢弃的奶酪，她想用这些奶酪喂鸟，但是奶酪的数量太多，人们认为这一假设不可能。“这件事情已经失去控制了，不可能是老奶奶干的，依靠退休金是买不起这么多奶酪的，”一名女子在社交媒体“脸谱”(Facebook)上写道，“也许是一名行为古怪、但是乐善好施的富豪干的。”</p><p>　　当地居民Rachel Cassar表示，她的狗狗在过去两周内发现了7块奶酪，其中一个就在她家院内。“有人在持续地往这里投放大量的奶酪，我的一位朋友已经给她家的狗洗胃了，因为它们吃了一块奶酪。现在这些奶酪很多都已经长毛发霉了，我的狗狗把它们堆在后院，但是有些奶酪看起来还新鲜，我的狗狗目前状况还好。”Rachel Cassar说。</p><p>　　另一位居民Emma 表示，她的狗狗Ollie几个月前就在花园里发现了一块奶酪，而他们家已经很多年没有买过奶酪了。</p><div id="function_code_page"></div>        </div>"""
    content=extract_html(content)
    print(content)

