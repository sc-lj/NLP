# coding:utf-8

from collections import namedtuple
import tensorflow as tf

FLAGS=tf.flags.FLAGS
tf.flags.DEFINE_string('data_path','', 'Path expression to tf.Example.')
tf.flags.DEFINE_string('vocab_path','data/chivocab', 'Path expression to text vocabulary file.')
tf.flags.DEFINE_string('article_key', 'article','tf.Example feature key for article.')
tf.flags.DEFINE_string('abstract_key', 'headline','tf.Example feature key for abstract.')
tf.flags.DEFINE_string('log_root', 'model/log', 'Directory for model root.')
tf.flags.DEFINE_string('train_dir', 'model/train', 'Directory for train.')
tf.flags.DEFINE_string('eval_dir', 'model/eval', 'Directory for eval.')
tf.flags.DEFINE_string('decode_dir', 'model/decode', 'Directory for decode summaries.')
tf.flags.DEFINE_string('mode', 'train', 'train/eval/decode mode')
tf.flags.DEFINE_integer('max_run_steps', 10000000,'Maximum number of run steps.')
tf.flags.DEFINE_integer('max_article_sentences', 2,'Max number of first sentences to use from the article')
tf.flags.DEFINE_integer('max_abstract_sentences', 100,'Max number of first sentences to use from the abstract')
tf.flags.DEFINE_integer('beam_size', 4,'beam size for beam search decoding.')
tf.flags.DEFINE_integer('eval_interval_secs', 60, 'How often to run eval.')
tf.flags.DEFINE_bool('use_bucketing', False,'Whether bucket articles of similar length.')
tf.flags.DEFINE_bool('truncate_input', False,'Truncate inputs that are too long. If False,  examples that are too long are discarded.')
tf.flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used.')
tf.flags.DEFINE_integer('random_seed', 111, 'A seed value for randomness.')





