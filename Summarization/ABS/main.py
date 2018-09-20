# coding:utf-8

import ABS,BatchReader,Decode
import tensorflow as tf
import Data

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
tf.flags.DEFINE_integer('checkpoint_secs',60,'how often to save model')


def _Train(model,batcher):
    with tf.device('/cpu:0'):
        model.build_graph()
        saver=tf.train.Saver()
        saver_hook=tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.log_root,saver=saver,save_secs=FLAGS.checkpoint_secs)
        summery_writer=tf.summary.FileWriter(FLAGS.train_dir)
        sess=tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.log_root,
                                             is_chief=True,
                                             hooks=[saver_hook],
                                             config=tf.ConfigProto(allow_soft_placement=True))
        step = 0
        while not sess.should_stop() and step<FLAGS.max_run_steps:
            (article_batch, abstract_batch, targets, article_lens, abstract_lens,
             loss_weights, _, _) = batcher.NextBatch()
            (_, summaries, loss, train_step) = model.run_train_step(
                sess, article_batch, abstract_batch, targets, article_lens,
                abstract_lens, loss_weights)

            summery_writer.add_summary(summaries,train_step)
        sess.Stop()

def main():
    batch_size=6
    if FLAGS.mode=='decode':
        batch_size=FLAGS.beam_size

    hps=ABS.HParams(batch_size=batch_size,
                    mode=FLAGS.mode,
                    num_softmax_samples=4056,
                    min_lr=0.01,
                    lr=0.9,
                    hid_dim=400,
                    emb_dim=200,
                    max_grad_norm=2,
                    C=5,
                    Q=2,
                    dec_timesteps=90,
                    enc_timesteps=520)
    vocab=Data.Vocab(FLAGS.data_path)
    batcher=BatchReader.Batcher(vocab,hps,max_article_sentences=FLAGS.max_article_sentences,
                                max_abstract_sentences=FLAGS.max_abstract_sentences,
                                )
    vocab_size=vocab.NumIds()
    tf.set_random_seed(FLAGS.random_seed)
    if hps.mode=='train':
        model=ABS.ABS(hps,vocab_size)
        _Train(model,batcher)
    elif hps.mode=='decode':
        newhps=hps._replace(dec_timesteps=hps.C)
        model=ABS.ABS(hps,vocab_size)
        decoder = Decode.BSDecoder(model, batcher, newhps, vocab)
        decoder.DecodeLoop()

if __name__ == '__main__':
    tf.app.run()

