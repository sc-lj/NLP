# coding:utf-8

import tensorflow as tf
from .config import *
import os,datetime

class TextCNN():
    FLAGS=seq_param()
    def __init__(self, vector_length,lable_length):
        """
        :param vector_length: 词向量的长度
        :param lable_length: 类别长度
        """
        
        self.filter_sizes=self.FLAGS.filter_sizes
        self.vector_length=vector_length
        self.lable_length=lable_length
        self.input_x=tf.placeholder(shape=[None,self.vector_length],name='input_x',dtype=tf.float32)
        self.input_y=tf.placeholder(shape=[None,self.lable_length],name='input_y',dtype=tf.float32)
    
    
    def weight(self,shape):
        """
        权重向量或者卷积向量
        :param shape:
        :return:
        """
        weight=tf.Variable(initial_value=tf.random_normal(shape,stddev=2.0),name='weight')
        return weight
    
    def biases(self,shape):
        biases=tf.Variable(initial_value=tf.random_normal(shape=shape,stddev=1.0),name='biases')
        return biases
    
    def create_model(self):
        with tf.name_scope('conv'):
            pools=[]
            for i,filter_size in enumerate(self.filter_sizes):
                # 由于conv2d卷积shape为[height, width, in_channels, out_channels]，
                # in_channels: 图片的深度；在文本处理中深度为1，需要添加的一个1，增加其维度。
                filtersize=[filter_size,self.vector_length,1,self.FLAGS.filter_nums]
                # 'SAME'模式下的convd的形状是：[1,sequence_length-filter_size+1,1,1]
                convd=tf.nn.conv2d(self.input_x,filter=self.weight(filtersize),strides=(1,1,1,1),padding='SAME',name='convd')
                biase_shape=[self.FLAGS.filter_nums]
                convd=tf.nn.relu(tf.nn.bias_add(convd,self.biases(biase_shape)),name='relu')
                # convd的shape为：[batch_size,1]
                pooled=tf.nn.max_pool(convd,ksize=(1,self.FLAGS.seqence_length-filter_size+1,1,1),strides=(1,1,1,1),padding='SAME',name='pool')
                pools.append(pooled)

            num_filters_total=self.FLAGS.filter_nums*len(self.filter_sizes)
            self.h_pool=tf.concat(pools,3)
            # pool_flat的shape为：[batch_size,num_filters_total]
            self.pool_flat=tf.reshape(self.h_pool,[-1,num_filters_total])

        # dropout layer随机地选择一些神经元
        with tf.name_scope('dropout'):
            self.h_drop=tf.nn.dropout(self.pool_flat,self.FLAGS.dropout_prob)

        with tf.name_scope('output'):
            W=self.weight([num_filters_total,self.lable_length])
            b=self.biases([self.lable_length])
            self.score=tf.nn.xw_plus_b(self.h_drop,W,b,name='score')
            self.prediction=tf.argmax(self.score,1,name='prediction')

        with tf.name_scope('loss'):
            losses=tf.nn.softmax_cross_entropy_with_logits(self.score,self.input_y)
            self.loss=tf.reduce_mean(losses)

        with tf.name_scope('accuracy'):
            correct_predictions=tf.equal(self.prediction,tf.argmax(self.input_y,1))
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions,'float'),name='accuracy')

    def train_model(self,x_train, y_train, vocab_processor, x_dev, y_dev):
        with tf.Graph().as_default():
            gpu_config=tf.GPUOptions(per_process_gpu_memory_fraction=self.FLAGS.per_process_gpu_memory_fraction)
            session_conf=tf.ConfigProto(gpu_options=gpu_config,
                                        allow_soft_placemen=self.FLAGS.allow_soft_placemen,
                                        log_device_placemen=self.FLAGS.log_device_placement)
            # 配置session参数
            sess=tf.Session(config=session_conf)
            with sess.as_default():
                #用于记录全局训练步骤的单值
                global_step=tf.Variable(0,name='global_step',trainable=False)
        
                # 定义优化算法
                optimizer=tf.train.AdamOptimizer(1e-4)

                # minimize 只是简单的结合了compute_gradients和apply_gradients两个过程，
                # 如果单独使用compute_gradients和apply_gradients可以组合自己的意愿处理梯度
                # train_op=optimizer.minimize(self.loss,global_step=global_step)
                
                grads_and_vars=optimizer.compute_gradients(self.loss)
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
                loss_summary=tf.summary.scalar('loss', self.loss)
                accuracy_summary=tf.summary.scalar('accuracy',self.accuracy)
                
                out_dir= self.FLAGS.out_dir
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
        
                saver=tf.train.Saver(tf.all_variables(),max_to_keep=self.FLAGS.num_checkpoints)
                
                # 写入词汇表
                vocab_processor.save(os.path.join(out_dir,'vocab'))
                
                sess.run(tf.global_variables_initializer())
                
                def train_step(x_batch, y_batch):
                    feed_dict={self.input_x:x_batch,
                                self.input_y:y_batch}
                    
                    
                    _,step,summary,loss,accuracy=sess.run([train_op,global_step,train_summary,self.loss,self.accuracy],feed_dict)

                    time_str=datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str,step,loss,accuracy))
                    
                    train_writer.add_summary(summary,step)
                    
                
                def dev_step(x_batch, y_batch, writer=None):
                    feed_dict={self.input_x:x_batch,self.input_y:y_batch}
    
                    step,summary,loss,accuracy=sess.run([global_step,dev_summary,self.loss,self.accuracy],
                                                          feed_dict)
    
                    time_str=datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str,step,loss,accuracy))
    
                    
                    if writer:
                        writer.add_summary(summary,step)
                        
                





