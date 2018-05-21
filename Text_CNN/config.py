# coding:utf-8

import tensorflow as tf

def seq_param():
    # 卷积的宽度，在cnn中，卷积的长度为词向量的长度，宽度表示包含多少词
    tf.flags.DEFINE_string('filter_size', '2,4,6', 'Comma-separated filter sizes (default: "3,4,5")')
    
    # 每种卷积的个数
    tf.flags.DEFINE_integer('filter_num', 128,  "Number of filters per filter size (default: 128)")

    # dropout概率
    tf.flags.DEFINE_float("dropout_prob",0.5,"Dropout keep probability (default: 0.5)")

    # 规定每个句子的长度
    tf.flags.DEFINE_integer("seqence_length", 600, "seqence max length (default: 600)")
    
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
    
    FLAGS=tf.flags.FLAGS
    return FLAGS
    
    
    




