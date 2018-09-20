import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
from nnsum.utils.rouge_score import RougeScorer
import multiprocessing as par
import math

class MRTModel:

  def __init__(self, refs_dict, model, num_samples = 20, budget = 100, alpha = 0.05, gamma = 1.0, stopwords = set()):
    self.model = model
    self.num_samples = num_samples
    self.budget = budget
    self.refs_dict = refs_dict
    self.alpha = alpha
    self.gamma = gamma
    self.avg_rouge = 0.0
    self.scorer = RougeScorer(stopwords, word_limit = budget)
    
  def __call__(self, inputs, metadata):
    return self.forward(inputs, metadata)

  def cuda(self, c):
    return self.model.cuda(c)

  def train(self):
    return self.model.train()

  def eval(self):
    return self.model.eval()
  
  def parameters(self):
    return self.model.parameters()

  def predict(self, inputs, metadata, return_indices=False,
                decoder_supervision=None, max_length=100):
    return self.model.predict(inputs=inputs, metadata=metadata, return_indices=return_indices,
                decoder_supervision=decoder_supervision, max_length=max_length)

  def get_risk(self, samples, ids, texts):
        """
        Selects the first *budget* sentences that are equal to 1 and computes
        the total risk for that selection.
        Returns a batch_size x sample_size tensor of risks for each sample.
        In this callback you would implement -1*Rouge for each sample.
        """
        '''batch_size = samples.size(0)
        params = map(lambda x: (x, samples, ids, texts, indices, self.scorer, self.refs_dict), list(range(batch_size)))
        results = self.pool.map(_get_risk, params)
        return Variable(torch.stack(results))
        '''

        batch_size = samples.size(0)
        sample_size = samples.size(1)
        seq_size = samples.size(2)
        sample_risk = samples.data.new(batch_size, sample_size).float().fill_(0)

        for b in range(batch_size):
            for s in range(sample_size):
                discarded = 0
                for sen in range(seq_size): 
                    if samples.data[b, s, sen] == 1 and sen < len(texts[b]):
                        if not self.scorer.update(texts[b][sen],"%s-%s" % (ids[b],sen)):
                          discarded += 1  
                # this is the computation of risk based on rouge
                rouge = self.scorer.compute(self.refs_dict[ids[b]])
                self.avg_rouge = 0 # self.avg_rouge * 0.99 + rouge * 0.01
                sample_risk[b, s] = -(rouge - self.avg_rouge)
        return Variable(sample_risk)

  def make_mask(self, lengths):
    max_len = lengths.data.max()
    mask = lengths.data.new(lengths.size(0), max_len).fill_(0)

    for i, l in enumerate(lengths.data.tolist()):
        if l < max_len:
            mask[i,l:].fill_(1)
    return Variable(mask.byte())

  def cudalong(self, v):
    return v.to(dtype=torch.long, device=torch.device("cuda"))

  def categorical(self, probs):
    batch = []
    for b in range(probs.size(0)):
      idx = self.cudalong(torch.multinomial(probs.data[b]+0.0001,3,replacement=False))
      samples = self.cudalong(torch.cuda.LongTensor(probs.size(1),probs.size(2)).fill_(0))
      samples = samples.scatter(1,idx,1)
      batch.append(samples)
    return torch.stack(batch)

  def forward(self, inputs, metadata):
    logits = self.model.forward(inputs)
    probs = torch.sigmoid(logits)
    
    lengths = inputs.num_sentences
    mask = self.make_mask(lengths)
    sample_mask = mask.unsqueeze(1).repeat(1, self.num_samples, 1)

    ### Compute the model distribution P ###

    # probs is a batch_size x num_samples x seq_size tensor.
    # probs = F.sigmoid(logits)
    assert(0.0 == (probs.data > 1.0).nonzero().sum())
    assert(0.0 == (probs.data < 0).nonzero().sum())
    probs = probs.unsqueeze(1).repeat(1, self.num_samples, 1)

    # overcomplete_probs is a batch_size x num_samples x seq_size x 2 tensor 
    # of probs where the last dim has probs of 0 and 1. Last dim sums to 1.
    # We need this to efficiently index the probabilities of each sentence
    # selection decision.
    probs4D = probs.unsqueeze(3)
    overcomplete_probs = torch.cat([1 - probs4D, probs4D], 3)

    # Samples is a batch_size x num_samples x seq_size tensor of 0s and 1s.
    # 1 indicates we are selecting the corresponding sentence to go in the
    # summary. samples is cast to a long tensor so we can use it to efficiently
    # index the probability of each decision so we can compute the likelihood
    # of each sequence under the model distribution P.
    #samples = torch.bernoulli(probs)
    #samples = Variable(samples.data.long())
    samples = self.categorical(probs)

    # Get probability of each sentence decision
    stepwise_sample_probs = overcomplete_probs.gather(
        dim=3, index=samples.unsqueeze(3)).squeeze(3)
    # Compute the sample log likelihood under the model distribution P:
    # log P(y|x) = sum logP(y_1|x) + logP(y_2|x) ...
    # sample_log_likelihood is a batch_size x sample_size tensor.
    sample_log_likelihood = torch.log(stepwise_sample_probs).masked_fill(
        sample_mask, 0).sum(2)
    
    # Length normalize the log likelihoods and take the softmax over each 
    # sample to get the approximate distribution Q.
    # For k samples:
    #
    #                       exp(logP(y^(i)|x)) 
    #  Q(y^(i)|x) =  -----------------------------------------
    #                  sum_j=1^k exp(logP(y^(j)|x))
    #
    q_logits = sample_log_likelihood / lengths.float().view(-1, 1) * self.alpha
    q_probs = F.softmax(q_logits, 1)

    # Here is where you would call your rouge function and return 1 - rouge
    # to make it a risk. sample_risks is a batch_size x num_samples tensor.
    sample_risks = self.get_risk(samples, metadata.id, metadata.texts)
    expected_risk = (q_probs * sample_risks).sum(1)
    # average the expected_risk over the batch
    avg_expected_risk = expected_risk.mean()
    return avg_expected_risk
