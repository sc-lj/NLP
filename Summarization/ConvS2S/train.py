# coding:utf-8

import tensorflow as tf
from convs2s_model import ConvS2SModel,HParams
from Data import *
from batch_reader import *
from Rouge import *

FLAGS=tf.flags.FLAGS
tf.flags.DEFINE_string('vocab_path','../data/chivocab', 'Path expression to text vocabulary file.')
tf.flags.DEFINE_string("topic_vocab_path","../data/topic_vocab.txt","Path expression to text topic vocabulary file")
tf.flags.DEFINE_string('log_root', './model/log', 'Directory for model root.')
tf.flags.DEFINE_string('train_dir', './model/train', 'Directory for train.')
tf.flags.DEFINE_string('eval_dir', './model/eval', 'Directory for eval.')
tf.flags.DEFINE_string('decode_dir', './model/decode', 'Directory for decode summaries.')
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
tf.flags.DEFINE_integer('checkpoint_secs',60,'how often to save model')
tf.flags.DEFINE_string("update_rule","adam","update model rule")


class ConS2S():
    def __init__(self,model,hps,batch_reader,vocab):
        self.model=model
        self._hps=hps
        self.batch_reader=batch_reader

        if FLAGS.update_rule=="Adam":
            self.optimizer=tf.train.AdamOptimizer
        elif FLAGS.update_rule=="Adagrad":
            self.optimizer=tf.train.AdagradOptimizer
        elif FLAGS.update_rule=="GD":
            self.optimizer=tf.train.GradientDescentOptimizer
        elif FLAGS.update_rule=='RMS':
            self.optimizer=tf.train.RMSPropOptimizer
        elif FLAGS.update_rule=='Ada':
            self.optimizer=tf.train.AdadeltaOptimizer
        else:
            raise("请提供以下几种优化算法，Adam,Adagrad,GD,RMS,Ada")

        self.word_to_topic=vocab.WordToTopic()



    def train(self):

        tf.get_variable_scope().reuse_variables()

        sampled_captions, _ = self.model._sample()

        greedy_caption = self.model._greed_sample()

        rewards = tf.placeholder(tf.float32, [None])
        base_line = tf.placeholder(tf.float32, [None])

        grad_mask = tf.placeholder(tf.int32, [None, 16])
        t1 = tf.expand_dims(grad_mask, 1)
        t1_mul = tf.to_float(tf.transpose(t1, [0, 2, 1]))

        loss = self.model._build_loss()

        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            norm = tf.reduce_sum(t1_mul)
            r  =  rewards - base_line
            sum_loss = - tf.reduce_sum(tf.transpose(tf.mul(tf.transpose(loss, [2, 1, 0]),r), [2, 1, 0]))/ norm
            grad_rl,_=tf.clip_by_global_norm(tf.gradients(sum_loss,tf.trainable_variables(),aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N),5.0)
            grads_and_vars=list(zip(grad_rl,tf.trainable_variables()))
            train_op=optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        config=tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth=True
        config.gpu_options.allocator_type="BFC"

        with tf.Session(config=config) as sess:
            saver=tf.train.Saver()
            meta=tf.train.get_checkpoint_state(FLAGS.log_root)
            if meta and meta.model_checkpoint_path:
                saver.restore(sess,meta.model_checkpoint_path)
            else:
                pass

            for e in range(FLAGS.max_run_steps):
                (enc_input_batch,enc_position_batch,enc_topic_batch,enc_lens,dec_input_batch,dec_lens,target_batch) = self.batch_reader.NextBatch()
                ref_decoded =target_batch
                feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}

                samples, greedy_words = sess.run([sampled_captions, greedy_caption],feed_dict)

                r = ''
                b = ''

                feed_dict = {grad_mask: mask, self.model.sample_caption:samples ,rewards: r, base_line: b,
                             self.model.features: features_batch, self.model.captions: captions_batch
                             }  # write summary for tensorboard visualization
                _ = sess.run([train_op], feed_dict)






