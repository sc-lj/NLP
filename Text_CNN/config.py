# coding:utf-8

import tensorflow as tf

def seq_param():
    # 卷积的宽度，在cnn中，卷积的长度为词向量的长度，宽度表示包含多少词
    tf.flags.DEFINE_string('filter_size', '2,4,6', 'Comma-separated filter sizes (default: "3,4,5")')
    
    # 每种卷积的个数
    tf.flags.DEFINE_integer('filter_num', 128,  "Number of filters per filter size (default: 128)")

    # dropout概率
    tf.flags.DEFINE_float("dropout_prob",0.5,"Dropout keep probability (default: 0.5)")

    # 训练多少次后，进行验证
    tf.flags.DEFINE_integer("evaluate_every", 100, "evaluate model on dev set after many step (default: 100)")
    
    # 是否记录设备指派情况
    tf.flags.DEFINE_boolean("log_device_placement",False,"Log placement of ops on devices")
    
    # 是否自动选择运行设备
    tf.flags.DEFINE_boolean("allow_soft_placement",True,"Allow device soft device placement")
    
    # 限制gpu的使用率，如果满负荷运转，会造成其他资源无法调用gpu
    tf.flags.DEFINE_float('per_process_gpu_memory_fraction', 0.6, "limit gpu use,other could use gpu")
    
    # 模型和summary的输出文件夹
    tf.flags.DEFINE_string('out_dir', './', 'Output directory for model and summary')
    
    # 最多保存的中间结果
    tf.flags.DEFINE_integer("num_checkpoints",5,"Number of checkpoints to store (default: 5)")

    # 每块包含的数据量
    tf.flags.DEFINE_integer('batch_size',60,'num of each batch')

    # 训练轮数
    tf.flags.DEFINE_integer('num_epochs',200,'Number of training epochs (default: 200)')

    # 验证集的比例
    tf.flags.DEFINE_float('dev_sample_percent',0.1,'percentage of the train data to use for evaluate')

    FLAGS=tf.flags.FLAGS
    return FLAGS
    
    
    




