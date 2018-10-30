# coding:utf-8
import tensorflow as tf
from collections import namedtuple

HParams=namedtuple("HParams","token_len label_len dropout_pro hidden_state_dimension lr clipValue max_to_keep")

class Entity():
    def __init__(self,dataset,hparams):
        self._hps=hparams
        self.dataset=dataset
        # 这些是一个语句一个语句输入的，而不是batch输入的
        self.input_token_indices=tf.placeholder(dtype=tf.int32,shape=[None],name="input_token_indices")
        self.input_label_indices=tf.placeholder(dtype=tf.int32,shape=[None],name="input_label_indices")
        self.input_label_indices_vector=tf.placeholder(dtype=tf.float32,shape=[None,dataset.num_of_class],name='input_label_indices_vector')
        self.input_token_length=tf.placeholder(dtype=tf.float32,shape=[None],name='input_token_length')

        # 返回一个用于初始化权重的初始化程序，这个初始化器是用来保持每一层的梯度大小都差不多相同
        self.initializer=tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("token_embedding"):
            self.token_embedding_weights=tf.get_variable(name='token_embedding_weights',shape=[dataset.vocab_size,self._hps.embeddim],dtype=tf.float32,initializer=self.initializer)
            embedding_token=tf.nn.embedding_lookup(self.token_embedding_weights,self.input_token_indices)

        with tf.variable_scope("dropout"):
            token_lstm_input_drop=tf.nn.dropout(embedding_token,self._hps.dropout_pro,name="token_lstm_input_drop")
            token_lstm_input_drop_expanded = tf.expand_dims(token_lstm_input_drop, axis=0, name='token_lstm_input_drop_expanded')


        with tf.variable_scope('token_lstm'):
            token_lstm_output=self.entity_lstm(token_lstm_input_drop_expanded,self.input_token_length,True)

        with tf.variable_scope('feedforward_after_lstm'):
            W=tf.get_variable(name='W',shape=[2*self._hps.hidden_state_dimension,self._hps.hidden_state_dimension],initializer=self.initializer)
            b=tf.get_variable(tf.constant(0.0,name='b',shape=[self._hps.hidden_state_dimension]),name='biase')
            outputs=tf.nn.xw_plus_b(token_lstm_output,W,b,name='output_before_tanh')
            outputs=tf.nn.tanh(outputs,name='output_after_tanh')

        with tf.variable_scope('feedforward_before_crf'):
            W=tf.get_variable("W",shape=[self._hps.hidden_state_dimension,dataset.num_of_class],initializer=self.initializer)
            b=tf.get_variable(tf.constant(0.0,dtype=tf.float32,shape=[dataset.num_of_class]),name='biase')
            scores=tf.nn.xw_plus_b(outputs,W,b)
            self.unary_scores = scores
            self.predictions=tf.argmax(scores,axis=1,name='prediction')

        with tf.variable_scope("crf"):
            small_score = -1000.0
            large_score = 0.0
            # 在nary_scores矩阵0维的前后加上一行，以及1维的后面添加两列
            seqence_length=tf.shape(self.unary_scores)[0]
            unary_scores_with_start_and_end=tf.concat([self.unary_scores,tf.tile(tf.constant(small_score,dtype=tf.float32,shape=[1,2]),[seqence_length,1])],axis=1)
            start_unary_scores=[[small_score]*dataset.num_of_class+[large_score,small_score]]
            end_unary_scores=[[small_score]*dataset.num_of_class+[small_score,large_score]]
            self.unary_scores=tf.concat([start_unary_scores,unary_scores_with_start_and_end,end_unary_scores],0)

            start_index=dataset.num_of_class
            end_index=dataset.num_of_class+1
            input_label_indices_flat_with_start_and_end=tf.concat([tf.constant(start_index,shape=[1]),self.input_label_indices,tf.constant(end_index,shape=[1])],0)

            seqence_length=tf.shape(self.unary_scores)[0]
            seqence_lengths=tf.expand_dims(seqence_length,0,name='sequence_lengths')
            unary_scores_expanded=tf.expand_dims(self.unary_scores,0,name='unary_scores_expanded')
            input_label_indices_flat_batch=tf.expand_dims(input_label_indices_flat_with_start_and_end,0,name='input_label_indices_flat_batch')

            self.transition_parameters=tf.get_variable("transitions",shape=[dataset.num_of_class+2,dataset.num_of_class+2],dtype=tf.float32,initializer=self.initializer)
            log_likelihood, _=tf.contrib.crf.crf_log_likelihood(unary_scores_expanded,input_label_indices_flat_batch,seqence_lengths,transition_parameters=self.transition_parameters)
            self.loss=-tf.reduce_mean(log_likelihood,name='cross_entropy_mean_loss')
            self.accuracy = tf.constant(1)

        self.global_step=tf.Variable(0,trainable=False,name='global_step')
        self.optimizer=tf.train.AdamOptimizer(self._hps.lr)
        grads_and_vars=self.optimizer.compute_gradients(self.loss)
        grads_and_vars=[(tf.clip_by_value(grads,-self._hps.clipValue,self._hps.clipValue),vars) for grads,vars in grads_and_vars]
        self.train_op=self.optimizer.apply_gradients(grads_and_vars)

        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=self._hps.max_to_keep)  # defaults to saving all variables


    def entity_lstm(self,inputs,sequence_length,output_sequence):
        hps=self._hps
        batch_size=tf.shape(sequence_length)[0]
        lstm_cell={}
        initial_state={}
        with tf.variable_scope("bilstm"):
            for direction in ["backward","forward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction]=tf.nn.rnn_cell.LSTMCell(hps.hidden_state_dimension,initializer=self.initializer)
                    init_input_state=tf.get_variable(name='initial_cell_state',shape=[1,hps.hidden_state_dimension],dtype=tf.float32,initializer=self.initializer)
                    init_output_state=tf.get_variable(name='init_output_state',shape=[1,hps.hidden_state_dimension])
                    c_state=tf.tile(init_input_state,tf.stack([batch_size,1]))
                    h_state=tf.tile(init_output_state,tf.stack())
                    initial_state[direction]=tf.nn.rnn_cell.LSTMStateTuple(c_state,h_state)

            outputs,states=tf.nn.bidirectional_dynamic_rnn(lstm_cell['forward'],lstm_cell['backward'],inputs,dtype=tf.float32,
                                                           initial_state_fw=initial_state['forward'],initial_state_bw=initial_state['backward'],
                                                           sequence_length=sequence_length)
            if output_sequence==True:
                # outputs_forwards shape [batch_size,token_len,hidden_state_dimension]
                outputs_forwards,outputs_backwards=outputs
                output=tf.concat([outputs_forwards,outputs_backwards],axis=2,name='output_sequence')
            else:
                final_states_forward, final_states_backward = states
                output=tf.concat([final_states_forward[1],final_states_backward[1]],axis=1,name='output')
            return output



