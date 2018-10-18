# coding:utf-8

import tensorflow as tf
from convs2s_model import ConvS2SModel,HParams
from Data import *
from batch_reader import *
from Rouge import Rouge

FLAGS=tf.flags.FLAGS
tf.flags.DEFINE_string('vocab_path','../data/chivocab', 'Path expression to text vocabulary file.')
tf.flags.DEFINE_string("topic_vocab_path","../data/topic_vocab.txt","Path expression to text topic vocabulary file")
tf.flags.DEFINE_string('log_root', './model/log', 'Directory for model root.')
tf.flags.DEFINE_string('train_dir', 'model/train', 'Directory for train.')
tf.flags.DEFINE_string('mode', 'train', 'train/eval/decode mode')
tf.flags.DEFINE_integer('max_run_steps', 10000000,'Maximum number of run steps.')
tf.flags.DEFINE_integer('max_article_sentences', 2,'Max number of first sentences to use from the article')
tf.flags.DEFINE_integer('max_abstract_sentences', 100,'Max number of first sentences to use from the abstract')
tf.flags.DEFINE_integer('beam_size', 4,'beam size for beam search decoding.')
tf.flags.DEFINE_integer('eval_interval_secs', 60, 'How often to run eval.')
tf.flags.DEFINE_bool('use_bucketing', False,'Whether bucket articles of similar length.')
tf.flags.DEFINE_bool('truncate_input', False,'Truncate inputs that are too long. If False,  examples that are too long are discarded.')
tf.flags.DEFINE_integer('random_seed', 111, 'A seed value for randomness.')
tf.flags.DEFINE_integer('checkpoint_secs',60,'how often to save model')

tf.flags.DEFINE_string("update_rule","Adam","update model rule")
tf.flags.DEFINE_float("lr",0.01,"learning rate")
tf.flags.DEFINE_string("rouge","rouge_l","Abstract summary evaluation method")

tf.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")#worker和ps
tf.flags.DEFINE_integer("task_index", 0, "Index of task within the job") #index有三个task_index，即0，1，2
tf.flags.DEFINE_integer("num_gpus", 1,"Total number of gpus for each machine. If you don't use GPU, please set it to '0'")

# 同步训练模型下，设置收集工作节点数量。默认工作节点总数
tf.flags.DEFINE_integer("replicas_to_aggregate", None,"Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: num_workers)")

# 使用同步训练、异步训练
tf.flags.DEFINE_boolean("sync_replicas", False,"Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated before applied to avoid stale gradients")
# 定义集群
# ps_hosts = ["xx.xxx.xx.xxxx:oooo", "xx.xxx.xx.xxxx:oooo"]  # 两个参数服务器
# worker_hosts = ["xx.xxx.xx.xxxx:oooo", "xx.xxx.xx.xxxx:oooo", "xx.xxx.xx.xxxx:oooo"]  # 两个计算服务器
# cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
# server = tf.train.Server(cluster,
#                          job_name=FLAGS.job_name,
#                          task_index=FLAGS.task_index)
# server.join()


class ConS2S():
    def __init__(self,model,hps,batch_reader,vocab):
        self.model=model
        self._hps=hps
        self.vocab=vocab
        self.batch_reader=batch_reader
        self.end_id=vocab.WordToId(PARAGRAPH_END)

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

        self.word_to_topic=np.array(vocab.WordToTopic())
        self.indicator=np.array((vocab.indicator))
        self.lr=FLAGS.lr
        self.rouge=FLAGS.rouge

    def train(self):
        # num_workers = len(worker_hosts)
        # tf.train.Server定义开始，每个节点就不一样了。根据执行的命令参数不同，决定了这个任务是哪个任务。
        # 如果任务名字是ps的话，程序就join到这里，作为参数更新的服务，等待其他worker节点给他提交参数更新的数据。
        # 如果是worker任务，就继续执行后面的计算任务。
        # server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)
        # if FLAGS.job_name == "ps":
        #     server.join()

        # 函数replica_deviec_setter会自动分配到参数服务器上去定义，如果有多个参数服务器，就轮流循环分配
        # with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,ps_device="/job:ps/cpu:0",cluster=cluster)):
        with tf.variable_scope("train") as scope:
            # 这一步是建立tensor流程图，至关重要
            self.model.build_graph()
            tf.get_variable_scope().reuse_variables()# 重用变量
            summaries = tf.summary.merge_all()
            global_step = tf.Variable(0,trainable=False,name="global_step")
            sampled_captions, _ = self.model._sample()
            greedy_caption = self.model._greed_sample()

            rewards = tf.placeholder(tf.float32, [None])
            base_line = tf.placeholder(tf.float32, [None])

            grad_mask = tf.placeholder(tf.int32, [None])
            t1_mul = tf.to_float(grad_mask)

            loss = self.model._build_loss()

            with tf.variable_scope('optimizer',reuse=tf.AUTO_REUSE):
                # 异步训练模式：自己计算完成梯度就去更新参数，不同副本之间不会去协调进度
                optimizer = self.optimizer(learning_rate=self.lr)
                # 同步训练模式
                if FLAGS.sync_replicas:
                    if FLAGS.replicas_to_aggregate is None:
                        replicas_to_aggregate = num_workers
                    else:
                        replicas_to_aggregate = FLAGS.replicas_to_aggregate
                    # 使用SyncReplicasOptimizer作优化器，并且是在图间复制情况下
                    # 在图内复制情况下将所有梯度平均
                    optimizer = tf.train.SyncReplicasOptimizer(
                        optimizer,
                        replicas_to_aggregate=replicas_to_aggregate,
                        total_num_replicas=num_workers,
                        name="mnist_sync_replicas")
                norm = tf.reduce_sum(t1_mul)
                r = rewards - base_line
                sum_loss = - tf.reduce_sum(tf.transpose(tf.multiply(tf.transpose(loss, [1, 0]),r), [1, 0]))/ norm
                grad_rl,_=tf.clip_by_global_norm(tf.gradients(sum_loss,tf.trainable_variables()),5.0)
                grads_and_vars=zip(grad_rl,tf.trainable_variables())
                train_op=optimizer.apply_gradients(grads_and_vars=grads_and_vars,global_step=tf.train.get_or_create_global_step())

            # allow_soft_placement能让tensorflow遇到无法用GPU跑的数据时，自动切换成CPU进行。
            # config=tf.ConfigProto(allow_soft_placement=True,device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])
            config=tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth=True # 程序按需申请内存
            config.gpu_options.allocator_type="BFC" # 使用BFC算法

            saver = tf.train.Saver()
            saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.log_root, saver=saver,save_secs=FLAGS.checkpoint_secs)

            # Train dir is different from log_root to avoid summary directory conflict with Supervisor.
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
            # tf.train.Supervisor可以简化编程,避免显示地实现restore操作
            # MonitoredTrainingSession已经初始化所有变量了，不用在tf.initialize_all_variables()
            sess = tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.log_root,
                                                     is_chief=True,
                                                     hooks=[saver_hook],
                                                     config=config)

            # init= tf.initialize_all_variables()
            # sess.run(init)
            # sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
            #                          logdir=FLAGS.log_root,
            #                          init_op=init,
            #                          summary_op=summaries,
            #                          saver=saver,
            #                          global_step=global_step,
            #                          save_model_secs=600)

            # sess=sv.managed_session(server.target,config=config)
            step=0
            while not sess.should_stop() and step < FLAGS.max_run_steps:
                (enc_input_batch,enc_position_batch,enc_lens,dec_input_batch,dec_lens,dec_position_batch,target_batch) = self.batch_reader.NextBatch()
                ref_decoded =target_batch
                feed_dict = {self.model.article: enc_input_batch, self.model.article_position: enc_position_batch,
                             self.model.abstract:dec_input_batch,self.model.topic_to_vocab:self.word_to_topic,
                             self.model.abstract_position:dec_position_batch,self.model.indicator:self.indicator}

                samples, greedy_words = sess.run([sampled_captions, greedy_caption],feed_dict)

                r_rouge=Rouge(samples,ref_decoded,self.end_id)
                b_rouge=Rouge(greedy_words,ref_decoded,self.end_id)
                mask,r = r_rouge(self.rouge)
                _,b = b_rouge(self.rouge)

                feed_dict = {grad_mask: mask, self.model.sample_caption:samples ,rewards: r, base_line: b,
                             self.model.article: enc_input_batch, self.model.article_position: enc_position_batch,
                             self.model.abstract: dec_input_batch, self.model.topic_to_vocab: self.word_to_topic,
                             self.model.abstract_position: dec_position_batch,self.model.indicator:self.indicator
                             }  # write summary for tensorboard visualization
                train_step,_ = sess.run([train_op,global_step], feed_dict)
                summary_writer.add_summary(summaries, train_step)
                step+=1

            # sv.stop()

    def eval(self):
        sampled_captions, _ = self.model._sample()
        # allow_soft_placement能让tensorflow遇到无法用GPU跑的数据时，自动切换成CPU进行。
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True  # 程序按需申请内存
        config.gpu_options.allocator_type = "BFC"  # 使用BFC算法
        with tf.Session(config=config) as sess:
            saver=tf.train.Saver()
            meta=tf.train.get_checkpoint_state(FLAGS.log_root)
            if meta and meta.model_checkpoint_path:
                saver.restore(sess,meta.model_checkpoint_path)
            else:
                raise ValueError("该路径没有模型文件")
            (enc_input_batch, enc_position_batch, enc_lens, dec_input_batch, dec_lens,target_batch) = self.batch_reader.NextBatch(batch=1)


            feed_dict = {self.model.article: enc_input_batch, self.model.article_position: enc_position_batch,
                        self.model.abstract: dec_input_batch, self.model.topic_to_vocab: self.word_to_topic}

            samples= sess.run(sampled_captions, feed_dict)
            words=Ids2Words(samples,self.vocab)
            print("生成文章摘要为：","".join(words))

def main(argv):
    vocab = Vocab(FLAGS.vocab_path, FLAGS.topic_vocab_path)
    # Check for presence of required special tokens.
    assert vocab.CheckVocab(PAD_TOKEN) > 0
    assert vocab.CheckVocab(UNKNOWN_TOKEN) >= 0
    assert vocab.CheckVocab(SENTENCE_START) > 0
    assert vocab.CheckVocab(SENTENCE_END) > 0

    hps=HParams(batch_size=6,
                enc_timesteps=520,
                dec_timesteps=90,
                emb_dim=256,
                con_layers=6,
                kernel_size=6,
                min_input_len=10)

    batcher = Batcher(vocab, hps, FLAGS.max_article_sentences,
        FLAGS.max_abstract_sentences, bucketing=FLAGS.use_bucketing,
        truncate_input=FLAGS.truncate_input)

    tf.set_random_seed(FLAGS.random_seed)

    model=ConvS2SModel(hps,vocab)

    cons2s=ConS2S(model,hps,batcher,vocab)
    cons2s.train()


if __name__ == '__main__':
    tf.app.run()

