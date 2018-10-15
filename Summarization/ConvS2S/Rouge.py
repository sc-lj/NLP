# coding:utf-8
"""自动文本摘要评测方法"""

import tensorflow as tf

class Rouge():
    def __init__(self,target,refer,end_id):
        """
        :param target: 参考摘要
        :param refer: 自动摘要
        :param end_id:
        """
        self.target=target
        self.refer=refer
        self.end_id=end_id
        self.batch_size=tf.shape(self.target)[0]
        # 参考摘要的长度
        m=tf.cast(tf.argmax(tf.cast(tf.equal(self.target,self.end_id),tf.float32),1),tf.int32)
        # 如果没有end_id,不对参考摘要进行截断
        self.m=tf.where(tf.equal(m,0),tf.fill([self.batch_size],tf.shape(self.target)[1]-1),m)
        # 自动摘要的长度
        n=tf.cast(tf.argmax(tf.cast(tf.equal(self.refer,self.end_id),tf.float32),1),tf.int32)
        self.n=tf.where(tf.equal(n,0),tf.fill([self.batch_size],tf.shape(self.refer)[1]-1),m)


    def __call__(self, rouge_type,n=2):
        if "rouge_l" ==rouge_type:
            return self.rouge_l()
        elif "rouge_n" ==rouge_type:
            return self.rouge_n(n=n)
        else:
            raise ValueError("请选择正确的摘要评测方法，目前提供了rouge_l和rouge_n两种方法。")


    def rouge_n(self,n=2):
        """rouge-n
        :return  [batch_size]
        """
        batch_size=self.batch_size
        k=0
        total_rouge=tf.Variable([])
        _,_,total_rouge=tf.while_loop(
            cond=lambda k,*_:k<batch_size,
            body=self.step_n,
            loop_vars=[k,n,total_rouge]
        )
        rouge = tf.reshape(total_rouge,shape=[-1,1])
        return rouge


    def step_n(self,k,n,total_rouge):
        """calculate rouge-n"""
        target,refer,m1,n1=self.target[k],self.refer[k],self.m[k],self.n[k]

        k1 = 0
        value=tf.constant([0.])
        table_size=(m1+1-n)*(n1+1-n)
        def loop_step(k1,value):
            i=tf.cast(k1%(n1+1-n),tf.int32)#自动摘要的index
            j=tf.cast((k1-i)/(n1+1-n),tf.int32)#参考摘要的index
            val=tf.cond(tf.logical_or(tf.equal(i,0),tf.equal(j,0)),
                        lambda :0,
                        lambda :tf.cond(tf.equal(target[j-1:j+n-1],refer[i-1:i+n-1]),
                                        lambda :1,
                                        lambda :0
                                        )
                        )
            value+=val
            return k1+1,value

        _,value=tf.while_loop(
            cond=lambda k,*_:k<table_size,
            body=loop_step,
            loop_vars=[k1,value]
        )
        m1=tf.cast(m1,tf.float32)
        value=value/m1
        total_rouge=tf.concat([total_rouge,value],0)
        return k,n,total_rouge


    def rouge_l(self):
        """
        Computes ROUGE-L (sentence level) of two text collections of sentences (as Tensors).
        R_lcs = LCS(X,Y)/m
        P_lcs = LCS(X,Y)/n
        F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
        where:
        X = candidate summary
        Y = reference summary
        m = length of candidate summary
        n = length of reference summary

        Args:
          hyp_t: Tensor containing sequences to be evaluated
          ref_t: Tensor containing references
          end_id: end token, used to truncate sequences before calculating ROUGE-L

        Returns:
          1-D Tensor. Average ROUGE-L values
        """
        batch_size=self.batch_size
        k = 0
        total_rouge = tf.Variable([])
        _, total_rouge = tf.while_loop(
            cond=lambda k, *_: k < batch_size,
            body=self._step,
            loop_vars=[k, total_rouge])
        # get average ROUGE-L values
        rouge = tf.reshape(total_rouge,shape=[-1,1])
        return rouge


    def _step(self,k, total_rouge):
        # calculate ROUGE-L for every element
        llcs = self._tf_len_lcs(self.target[k], self.refer[k], self.m[k], self.n[k])
        res = self._f_p_r_lcs(llcs, self.m[k], self.n[k])
        res= tf.reshape(res, shape=[1])
        res = tf.cast(res, tf.float32)
        total_rouge =tf.concat([total_rouge,res],0)
        return k + 1, total_rouge


    def _tf_len_lcs(self,x, y, m, n):
        table = self._tf_lcs(x, y, m, n)
        return table[m, n]


    def _tf_lcs(self,x, y, m, n):
        table_size = (m + 1) * (n + 1)
        k = 0
        table = tf.TensorArray(tf.int32, table_size, clear_after_read=False, element_shape=[])

        def loop_step(k, table):
            j = tf.cast(k % (n + 1), tf.int32)
            i = tf.cast((k - j) / (n + 1), tf.int32)
            # 获取和写入当前的index的值
            val = tf.cond(tf.logical_or(tf.equal(i, 0), tf.equal(j, 0)),
                          lambda: 0,
                          lambda: tf.cond(tf.equal(x[i - 1], y[j - 1]),
                                          lambda: table.read((i - 1) * (n + 1) + j - 1) + 1,
                                          lambda: tf.maximum(table.read((i - 1) * (n + 1) + j),
                                                             table.read(i * (n + 1) + j - 1))))
            table = table.write(k, val)
            return k + 1, table

        # 循环
        _, table = tf.while_loop(
            cond=lambda k, *_: k < table_size,
            body=loop_step,
            loop_vars=[k, table])
        table = tf.reshape(table.stack(), [m + 1, n + 1])
        return table


    def _f_p_r_lcs(self,llcs,m,n):
        r_lcs = llcs / m
        p_lcs = llcs / n
        beta = p_lcs / (r_lcs + 1e-12)
        num = (1 + (beta**2)) * r_lcs * p_lcs
        denom = r_lcs + ((beta**2) * p_lcs)
        f_lcs = num / (denom + 1e-12)
        return f_lcs


