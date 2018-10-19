#coding:utf-8
import sys
import time

import tensorflow as tf
import batch_reader
import Data
import seq2seq_attention_decode
import seq2seq_attention_model

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('data_path','', 'Path expression to tf.Example.')
tf.flags.DEFINE_string('vocab_path','../data/chivocab', 'Path expression to text vocabulary file.')
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
tf.flags.DEFINE_integer('checkpoint_secs',60,'how often to save model')


def _RunningAvgLoss(loss, running_avg_loss, summary_writer, step, decay=0.999):
  """Calculate the running average of losses."""
  if running_avg_loss == 0:
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)
  loss_sum = tf.Summary()
  loss_sum.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  sys.stdout.write('running_avg_loss: %f\n' % running_avg_loss)
  return running_avg_loss


def _Train(model, data_batcher):
  """Runs model training."""
  with tf.device('/cpu:0'):
    model.build_graph()
    saver = tf.train.Saver()
    saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.log_root, saver=saver,
                                              save_secs=FLAGS.checkpoint_secs)
    # Train dir is different from log_root to avoid summary directory
    # conflict with Supervisor.
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
    # tf.train.Supervisor可以简化编程,避免显示地实现restore操作
    sess = tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.log_root,
                                             is_chief=True,
                                             hooks=[saver_hook],
                                             config=tf.ConfigProto(allow_soft_placement=True))
    running_avg_loss = 0
    step = 0
    while not sess.should_stop() and step < FLAGS.max_run_steps:
      (article_batch, abstract_batch, targets, article_lens, abstract_lens,
       loss_weights, _, _) = data_batcher.NextBatch()
      (_, summaries, loss, train_step) = model.run_train_step(
          sess, article_batch, abstract_batch, targets, article_lens,
          abstract_lens, loss_weights)

      summary_writer.add_summary(summaries, train_step)
      running_avg_loss = _RunningAvgLoss(loss,running_avg_loss, summary_writer, train_step)
      step += 1
      if step % 100 == 0:
        summary_writer.flush()
    sess.Stop()
    return running_avg_loss


def _Eval(model, data_batcher, vocab=None):
  """Runs model eval."""
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  running_avg_loss = 0
  step = 0
  while True:
    time.sleep(FLAGS.eval_interval_secs)
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue

    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.train_dir)
      continue

    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    (article_batch, abstract_batch, targets, article_lens, abstract_lens,
     loss_weights, _, _) = data_batcher.NextBatch()
    (summaries, loss, train_step) = model.run_eval_step(
        sess, article_batch, abstract_batch, targets, article_lens,
        abstract_lens, loss_weights)
    tf.logging.info(
        'article:  %s',
        ' '.join(Data.Ids2Words(article_batch[0][:].tolist(), vocab)))
    tf.logging.info(
        'abstract: %s',
        ' '.join(Data.Ids2Words(abstract_batch[0][:].tolist(), vocab)))

    summary_writer.add_summary(summaries, train_step)
    running_avg_loss = _RunningAvgLoss(
        running_avg_loss, loss, summary_writer, train_step)
    if step % 100 == 0:
      summary_writer.flush()


def main(unused_argv):
  vocab = Data.Vocab(FLAGS.vocab_path, 1000000)
  # Check for presence of required special tokens.
  assert vocab.CheckVocab(Data.PAD_TOKEN) > 0
  assert vocab.CheckVocab(Data.UNKNOWN_TOKEN) >= 0
  assert vocab.CheckVocab(Data.SENTENCE_START) > 0
  assert vocab.CheckVocab(Data.SENTENCE_END) > 0

  batch_size = 4
  if FLAGS.mode == 'decode':
    batch_size = FLAGS.beam_size

  hps = seq2seq_attention_model.HParams(
      mode=FLAGS.mode,  # train, eval, decode
      min_lr=0.01,  # min learning rate.
      lr=0.15,  # learning rate
      batch_size=batch_size,
      enc_layers=4,
      enc_timesteps=520, #文章的最长长度
      dec_timesteps=90, # 摘要的最长长度
      min_input_len=2,  # discard articles/summaries < than this
      num_hidden=256,  # for rnn cell
      emb_dim=128,  # If 0, don't use embedding
      max_grad_norm=2,
      num_softmax_samples=4096)  # If 0, no sampled softmax.

  batcher = batch_reader.Batcher(
      FLAGS.data_path, vocab, hps, FLAGS.article_key,
      FLAGS.abstract_key, FLAGS.max_article_sentences,
      FLAGS.max_abstract_sentences, bucketing=FLAGS.use_bucketing,
      truncate_input=FLAGS.truncate_input)
  tf.set_random_seed(FLAGS.random_seed)

  if hps.mode == 'train':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=FLAGS.num_gpus)
    _Train(model, batcher)
  elif hps.mode == 'eval':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=FLAGS.num_gpus)
    _Eval(model, batcher, vocab=vocab)
  elif hps.mode == 'decode':
    decode_mdl_hps = hps
    # Only need to restore the 1st step and reuse it since
    # we keep and feed in state for each step's output.
    decode_mdl_hps = hps._replace(dec_timesteps=1)
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        decode_mdl_hps, vocab, num_gpus=FLAGS.num_gpus)
    decoder = seq2seq_attention_decode.BSDecoder(model, batcher, hps, vocab)
    decoder.DecodeLoop()


if __name__ == '__main__':
  tf.app.run()
