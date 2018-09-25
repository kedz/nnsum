from nnsum.utils.word_tokenize import tokenize,normalize

class RougeScorer:

  def __init__(self, stopwords=set(),word_limit=100):
    self.stopwords = stopwords
    self.cache = dict()
    self.counts = dict()
    self.count = 0
    self.word_limit = word_limit

  def count_words(self, sen):
    c = 0
    d = dict()
    tokens = normalize(sen).split(" ")
    for tok in tokens:
      c += 1
      if tok not in self.stopwords:
        if tok not in d: 
          d[tok] = 1
        else:
          d[tok] += 1
    return d,c

  def sum2dict(self, d1, d2):
    for k in d1: 
      if k not in d2:
        d2[k] = d1[k]
      else: 
        d2[k] = d2[k] + d1[k]

  def update(self, sen, id, force=False):
    if id in self.cache:
      d,c = self.cache[id]
    else:
      d,c = self.count_words(sen)
      self.cache[id] = (d,c)
    if self.count + c <= self.word_limit or force:
      self.count += c
      self.sum2dict(d, self.counts)
      return True
    else:
      return False
    
  def compute(self, ref_dict):
    covered = 0
    ref_count = 0
    for key in ref_dict:
      ref_count += ref_dict[key]
      c = self.counts[key] if key in self.counts else 0
      covered += min(c,ref_dict[key])
    score = float(covered) / ref_count if ref_count >= 5 else 0
    self.counts = dict()
    self.count = 0
    return score
  
