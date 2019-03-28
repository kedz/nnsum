from .labels_mle_trainer import labels_mle_trainer
from torch.autograd import Variable

import torch
import torch.nn.functional as F

from ignite.engine import Engine, Events

def labels_mrt_trainer(model, optimizer, train_loader,
                       validation_loader, scorer, refs_dict, alpha=0.05, num_samples=100, 
                       max_epochs=10, pos_weight=None,
                       summary_length=100, remove_stopwords=True,
                       grad_clip=5, gpu=-1, model_path=None, 
                       results_path=None, teacher_forcing=-1,
                       create_trainer_fn=None,
                       valid_metric="rouge-2"):
    model.avg_rouge = 0.0
    model.scorer = scorer
    model.refs_dict = refs_dict
    model.alpha = alpha
    model.num_samples = num_samples

    labels_mle_trainer(model, optimizer, train_loader, validation_loader,
                       max_epochs=max_epochs, pos_weight=pos_weight,
                       summary_length=summary_length,
                       remove_stopwords=remove_stopwords, grad_clip=grad_clip,
                       gpu=gpu, model_path=model_path,
                       results_path=results_path,
                       teacher_forcing=teacher_forcing,
                       create_trainer_fn=create_trainer,
                       valid_metric=valid_metric)

def create_trainer(model, optimizer, pos_weight=None, grad_clip=5, gpu=-1):

    def _update(engine, batch):
        model.train()
        batch = batch.to(gpu)
        optimizer.zero_grad()
        logits = model(batch, decoder_supervision=batch.targets.float())
        mask = batch.targets.gt(-1).float()
        total_sentences_batch = int(batch.num_sentences.data.sum())
        
        if pos_weight is not None:
            mask.data.masked_fill_(batch.targets.data.eq(1), pos_weight)

        avg_risk = forward(logits, batch, model)
        avg_risk.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-grad_clip, grad_clip)
        optimizer.step()

        return {"total_xent": avg_risk*total_sentences_batch, 
                "total_examples": total_sentences_batch}

    trainer = Engine(_update)
   
    return trainer

def get_risk(samples, ids, texts, model):
        """
        Selects the first *budget* sentences that are equal to 1 and computes
        the total risk for that selection.
        Returns a batch_size x sample_size tensor of risks for each sample.
        In this callback you would implement -1*Rouge for each sample.
        """

        batch_size = samples.size(0)
        sample_size = samples.size(1)
        seq_size = samples.size(2)
        sample_risk = samples.data.new(batch_size, sample_size).float().fill_(0)

        for b in range(batch_size):
            for s in range(sample_size):
                for sen in range(seq_size):
                    if samples.data[b, s, sen] == 1 and sen < len(texts[b]):
                        if not model.scorer.update(texts[b][sen],"%s-%s" % (ids[b],sen)):
                            break
                # this is the computation of risk based on rouge
                rouge = model.scorer.compute(model.refs_dict[ids[b]])
                model.avg_rouge = 0 #model.avg_rouge * 0.999 + rouge * 0.001
                sample_risk[b, s] = -(rouge - model.avg_rouge)
        return Variable(sample_risk)

def make_mask(lengths):
    max_len = lengths.data.max()
    mask = lengths.data.new(lengths.size(0), max_len).fill_(0)

    for i, l in enumerate(lengths.data.tolist()):
        if l < max_len:
            mask[i,l:].fill_(1)
    return Variable(mask.byte())

def cudalong(v):
    return v.to(dtype=torch.long, device=torch.device("cuda"))

def categorical(probs):
    batch = []
    for b in range(probs.size(0)):
      idx = cudalong(torch.multinomial(probs.data[b]+0.0001,4,replacement=False))
      samples = cudalong(torch.cuda.LongTensor(probs.size(1),probs.size(2)).fill_(0))
      samples = samples.scatter(1,idx,1)
      batch.append(samples)
    return torch.stack(batch)

def forward(logits, batch, model):

    probs = torch.sigmoid(logits)

    lengths = batch.num_sentences
    mask = make_mask(lengths)
    sample_mask = mask.unsqueeze(1).repeat(1, model.num_samples, 1)

    ### Compute the model distribution P ###

    # probs is a batch_size x num_samples x seq_size tensor.
    # probs = F.sigmoid(logits)
    assert(0.0 == (probs.data > 1.0).nonzero().sum())
    assert(0.0 == (probs.data < 0).nonzero().sum())
    probs = probs.unsqueeze(1).repeat(1, model.num_samples, 1)

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
    samples = categorical(probs)

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
    q_logits = sample_log_likelihood / lengths.float().view(-1, 1) * model.alpha
    q_probs = F.softmax(q_logits, 1)

    # Here is where you would call your rouge function and return 1 - rouge
    # to make it a risk. sample_risks is a batch_size x num_samples tensor.
    sample_risks = get_risk(samples, batch.id, batch.sentence_texts, model)
    expected_risk = (q_probs * sample_risks).sum(1)
    # average the expected_risk over the batch
    avg_expected_risk = expected_risk.mean()
    return avg_expected_risk
