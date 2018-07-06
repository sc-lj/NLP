# coding:utf-8

import sys,os
sys.path.append(os.path.dirname(__file__))

import tensorflow as tf
from config import *
import datetime
from datasets import *
import gc

FLAGS = Argparse()
logger=log_config(__file__)
dealdata = DealData(FLAGS,logger)
class TextCNN():
    def __init__(self,bow_seq='seq'):
        """
        :param bow_seq:使用seq-cnn模型还是bow-cnn模型
        """
        self.FLAGS = FLAGS
        self.dealdata = dealdata
        self.max_sequence_length=self.dealdata.max_sequence_length#最大句子长度
        self.vector_length = self.dealdata.vocab_length#向量长度
        self.lable_length = len(self.dealdata.labels)#标签长度

        if bow_seq not in ['seq','bow']:
            assert '请选择seq模型或者bow模型中的其中一种'
        self.bow_seq=bow_seq
        self.filter_sizes=self.FLAGS.filter_size

        self.input_x=tf.placeholder(shape=[None,self.max_sequence_length,self.vector_length],name='input_x',dtype=tf.float32)
        self.input_y=tf.placeholder(shape=[None,self.lable_length],name='input_y',dtype=tf.float32)

        self.create_model()

    def weights(self,shape):
        """
        权重向量或者卷积向量
        :param shape:
        :return:
        """
        weight=tf.Variable(initial_value=tf.truncated_normal(shape,stddev=0.1,dtype=tf.float32),name='weight')
        return weight

    def biaseses(self,shape):
        biases=tf.Variable(initial_value=tf.constant(0.1,shape=shape,dtype=tf.float32),name='biases')
        return biases

    def create_model(self):
        pools=[]
        # 将input_x扩展维度
        input_x=tf.expand_dims(self.input_x,-1)
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-%s'%filter_size):
                # 由于conv2d卷积shape为[height, width, in_channels, out_channels]，
                # in_channels: 图片的深度；在文本处理中深度为1，需要添加的一个1，增加其维度。
                filtersize=[filter_size,self.vector_length,1,self.FLAGS.filter_num]
                # 'SAME'模式下的convd的形状是：[1,sequence_length-filter_size+1,1,1]
                convd=tf.nn.conv2d(input_x,self.weights(filtersize),strides=[1,1,1,1],padding='VALID',name='convd')
                biase_shape=[self.FLAGS.filter_num]
                convd=tf.nn.relu(tf.nn.bias_add(convd,self.biaseses(biase_shape)),name='relu')
                # convd的shape为：[batch_size,1]
                pooled=tf.nn.max_pool(convd,ksize=(1,self.max_sequence_length-filter_size+1,1,1),strides=[1,1,1,1],padding='VALID',name='pool')
                pools.append(pooled)

        num_filters_total=self.FLAGS.filter_num*len(self.filter_sizes)
        h_pool=tf.concat(pools,3)
        # pool_flat的shape为：[batch_size,num_filters_total]
        pool=tf.reshape(h_pool,[-1,num_filters_total])

        if self.FLAGS.is_training:
            # dropout layer随机地选择一些神经元
            with tf.name_scope('dropout'):
                dropout_prob=tf.constant(self.FLAGS.dropout_prob,dtype=tf.float32)
                pool=tf.nn.dropout(pool,dropout_prob)

        with tf.name_scope('output'):
            W=tf.get_variable('weight',shape=[num_filters_total,self.lable_length],initializer=tf.contrib.layers.xavier_initializer())
            b=self.biaseses([self.lable_length])
            self.score=tf.nn.xw_plus_b(pool,W,b,name='score')
            self.prediction=tf.argmax(self.score,1,name='prediction')

        with tf.name_scope('loss'):
            losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.score,labels=self.input_y)
            self.loss=tf.reduce_mean(losses)

        with tf.name_scope('accuracy'):
            correct_predictions=tf.equal(self.prediction,tf.argmax(self.input_y,1))
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions,'float'),name='accuracy')

def train_model(bow_seq='seq'):
    with tf.Graph().as_default():
        gpu_config=tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.per_process_gpu_memory_fraction)
        session_conf=tf.ConfigProto(#gpu_options=gpu_config,
                                    allow_soft_placement=FLAGS.allow_soft_placement,
                                    log_device_placement=FLAGS.log_device_placement)
        # 配置session参数
        sess=tf.Session(config=session_conf)

        with sess.as_default():
            """TextCNN必须在sess.as_default()下面执行，这是因为tensorflow在运行时会自动创建会话，但是不会创建默认的会话，即sess.as_default()。
            如果在其上面执行，那么TextCNN所在的会话与下面的程序不在同一个会话中，就会报错。
            如果不自己创建默认会话sess.as_default()，那么可以用with tf.Session(config=session_conf)代替，那么TextCNN(bow_seq=bow_seq)函数就可以在上面进行初始化了
            """
            cnn=TextCNN(bow_seq=bow_seq)
            #用于记录全局训练步骤的单值
            global_step=tf.Variable(0,name='global_step',trainable=False)

            # 定义优化算法
            optimizer=tf.train.AdamOptimizer(1e-3)

            # minimize 只是简单的结合了compute_gradients和apply_gradients两个过程，
            # 如果单独使用compute_gradients和apply_gradients可以组合自己的意愿处理梯度
            # train_op=optimizer.minimize(self.loss,global_step=global_step)

            grads_and_vars=optimizer.compute_gradients(cnn.loss)
            # 在参数上进行梯度更新,每执行一次 train_op 就是一次训练步骤
            train_op=optimizer.apply_gradients(grads_and_vars,global_step)

            # 记录梯度值和稀疏性
            grad_summaries=[]
            for g,v in grads_and_vars:
                if g is not None:
                    # 生成直方图
                    grads_hist_summary=tf.summary.histogram('{}/grad/hist'.format(v.name), g)
                    grad_summaries.append(grads_hist_summary)
                    #
                    sparsity_summary=tf.summary.scalar('{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(sparsity_summary)

            grad_summary_merge=tf.summary.merge(grad_summaries)

            # 记录loss和accuracy变化情况
            loss_summary=tf.summary.scalar('loss', cnn.loss)
            accuracy_summary=tf.summary.scalar('accuracy',cnn.accuracy)

            out_dir= FLAGS.out_dir
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # 训练的summary
            train_summary=tf.summary.merge([loss_summary,accuracy_summary,grad_summary_merge])
            train_summary_dir=os.path.join(out_dir, 'summary', 'train')
            train_writer=tf.summary.FileWriter(train_summary_dir,sess.graph)

            # 测试的summary
            dev_summary=tf.summary.merge([loss_summary,accuracy_summary])
            dev_summary_dir=os.path.join(out_dir, 'summary', 'dev')
            dev_writer=tf.summary.FileWriter(dev_summary_dir,sess.graph)



            checkpoint_dir = os.path.abspath(os.path.join(out_dir,'checkpoints'))

            checkpoint_prefix = os.path.join(checkpoint_dir,'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver=tf.train.Saver(tf.global_variables(),max_to_keep=FLAGS.num_checkpoints)

            sess.run(tf.global_variables_initializer())
            def train_step(x_batch, y_batch):
                feed_dict={cnn.input_x:x_batch,cnn.input_y:y_batch}

                _,step,summary,loss,accuracy=sess.run([train_op,global_step,train_summary,cnn.loss,cnn.accuracy],feed_dict)

                time_str=datetime.datetime.now().isoformat()
                print("{}:train step {}, loss {:g}, acc {:g}".format(time_str,step,loss,accuracy))
                train_writer.add_summary(summary,step)


            def dev_step(x_batch, y_batch, writer=None):
                feed_dict={cnn.input_x:x_batch,cnn.input_y:y_batch}
                step,summary,loss,accuracy=sess.run([global_step,dev_summary,cnn.loss,cnn.accuracy],
                                                      feed_dict)
                time_str=datetime.datetime.now().isoformat()
                print("{}:dev step {}, loss {:g}, acc {:g}".format(time_str,step,loss,accuracy))
                if writer:
                    writer.add_summary(summary,step)

            for x_batch, y_batch  in dealdata.batch_iter(bow_seq=bow_seq):
                train_step(x_batch, y_batch)
                current_step=tf.train.global_step(sess,global_step)
                if current_step % FLAGS.evaluate_every==0:
                    valid_data=dealdata.read_batch(bow_seq=bow_seq,batch_size=1000)
                    x_dev_vector, y_dev_array=valid_data.__next__()
                    """
                    总共有40000个测试样本，每次取出1000个测试样本进行测试
                    """
                    print("\nEvaluation:")
                    dev_step(x_dev_vector, y_dev_array, writer=dev_writer)
                    path = saver.save(sess, save_path=checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    del valid_data
                    gc.collect()





if __name__ == '__main__':
    train_model()





