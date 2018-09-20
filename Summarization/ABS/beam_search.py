# coding:utf-8

import numpy as np
class Hypothesis(object):
  """Defines a hypothesis during beam search."""

  def __init__(self, tokens, log_prob):
    """Hypothesis constructor.

    Args:
      tokens: start tokens for decoding.
      log_prob: log prob of the start tokens, usually 1.
      state: decoder initial states.
    """
    self.tokens = tokens
    self.log_prob = log_prob

  def Extend(self, token, log_prob):
    """Extend the hypothesis with result from latest step.

    Args:
      token: latest token from decoding.
      log_prob: log prob of the latest decoded tokens.
      new_state: decoder output state. Fed to the decoder for next step.
    Returns:
      New Hypothesis with the results from latest step.
    """
    return Hypothesis(self.tokens + [token], self.log_prob + log_prob)

  def latest_token(self,c):
    return self.tokens[-c:]

class Beam():
    def __init__(self,model,beam_size,start_token,end_token,max_steps):
        self._model=model
        self.beam_size=beam_size
        self.start_token=start_token
        self.end_token=end_token
        self.max_steps=max_steps

    def BeamSearch(self,sess,enc_inputs,enc_length,hps):
        """"""
        hyps=[Hypothesis(self.start_token,0.0)]*self.beam_size
        results=[]

        steps=0
        while steps<self.max_steps and len(results)<self.beam_size:
            last_token=[h.latest_token(hps.C) for h in hyps]
            token_len=[len(token) for token in last_token]
            weight=np.ones(max(token_len),dtype=np.int32)
            for i in range(len(token_len)):
                for j in range(token_len[i]):
                    weight[i][j]=1

            ids, prob=self._model.topk(sess,enc_inputs,enc_length,last_token,weight)

            all_hyps=[]
            num_beam_source=1 if steps==0 else len(hyps)
            for i in range(num_beam_source):
                h=hyps[i]
                for j in range(2*hps.batch_size):
                    all_hyps.append(h.Extend(ids[i,j],prob[i,j]))

            hyps=[]
            for h in self._BestHyps(all_hyps):
                if h.latest_token(hps.C)[-1]==self.end_token:
                    results.append(h)
                else:
                    hyps.append(h)
                if len(hyps)==self.beam_size or len(results)==self.beam_size:
                    break
            steps+=1

        if steps==self.max_steps:
            results.extend(hyps)

        return self._BestHyps(results)

    def _BestHyps(self,hybs):
        return sorted(hybs,key=lambda h: h.log_prob,reverse=True)


