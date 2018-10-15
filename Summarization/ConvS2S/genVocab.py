#coding:utf-8
"""Example of Converting TextSum model data.
Usage:
python genVocab.py --command binary_to_text --in_file data/data --out_file data/text_data
python genVocab.py --command text_to_binary --in_file data/text_data --out_file data/binary_data
python genVocab.py --command binary_to_text --in_file data/binary_data --out_file data/text_data2
diff data/text_data2 data/text_data
"""

import struct
import sys,re,jieba
from collections import defaultdict
import pandas as pd
import tensorflow as tf
from tensorflow.core.example import example_pb2
from database import MySQL
import Data,time

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('command', 'binary_to_text',
                           'Either binary_to_text or text_to_binary.'
                           'Specify FLAGS.in_file accordingly.')
tf.flags.DEFINE_string('in_file', 'data/data', 'path to file')
tf.flags.DEFINE_string('out_file', 'data/text_data', 'path to file')

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


rec=re.compile("[a-zA-Z]")
rec0=re.compile("[^\u4e00-\u9fa5a-zA-Z0-9<>/#.+_]")

def genVocab(vocabfile):
    mysql=MySQL()
    mysql.login()
    cursor=mysql.get_cursor()

    vocab=defaultdict(int)
    def imdict(ab):
        for a in ab.split(" "):
            a=a.strip()
            # 去掉全是小写的英文单词
            if len(a)==0 or (rec.match(a) and a.islower()) or (rec0.match(a)):
                continue
            vocab[a]+=1

    urlset=set()
    dalist = []
    tables=["news","crawldata"]
    for table in tables:
        sent="select title,brief,content,url from %s where 1"%table
        cursor.execute(sent)

        for title, brief, content,url in cursor.fetchall():
            if url in urlset:
                continue
            else:
                urlset.add(url)
            title = Data.extract_html(title,False)
            imdict(title)

            if table=="news" and brief is not None:
                brief= re.sub("摘要：","",brief)
                brief = Data.extract_html(brief,False)
                imdict(brief)
                brieflen=len(brief)
            else:brieflen=0

            content=re.sub("资料图（图源：.*?）","",content)
            try:
               content=Data.extract_html(content)
            except:
                continue
            time.sleep(0.1)
            imdict(content)
            contentlen=len(content)
            dalist.append([brieflen,contentlen])

    # data = pd.DataFrame(columns=["brief", "content"],data=dalist)
    # data=data[data['brief']>0]
    # data.to_csv("../data/len.csv",index=False)
    mysql.close()
    newvocab={Data.UNKNOWN_TOKEN:0,Data.PAD_TOKEN:-1,Data.SENTENCE_START:-1,Data.SENTENCE_END:-1}
    for key, value in vocab.items():
        if value >= 5:
            newvocab.update({key:value})
        else:
            newvocab[Data.UNKNOWN_TOKEN]+=value
    with open(vocabfile,'w') as f:
        for word,num in newvocab.items():
            f.write(word+" "+str(num)+"\n")


if __name__ == '__main__':
    # tf.app.run()
    genVocab("../data/chivocab")



