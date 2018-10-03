from threading import Thread
import multiprocessing
from nnsum.utils.word_tokenize import tokenize,normalize

def _get_refs(ids2refs, stopwords, word_limit, refs_dict, ids):
  for id in ids:
    sens = [normalize(line.strip()).split(" ") for line in open(ids2refs[id][0]).readlines()]
    tokens = [i for s in sens for i in s][:word_limit]
    d = dict()
    refs_dict[id] = d
    for t in tokens:
      if t in stopwords: continue
      if t not in d:
        d[t] = 1
      else:
        d[t] += 1
    if len(refs_dict) % (int(len(ids2refs)/3)) == 0 or len(refs_dict) == len(ids2refs):
      print("loading refs, done with %d from %d" % (len(refs_dict), len(ids2refs)))

def get_refs_dict(ids2refs, stopwords, word_limit):
  refs_dict = dict()
  threads = []
  keys = list(ids2refs.keys())
  step = int(len(ids2refs) / multiprocessing.cpu_count()) + 1
  for i in range(multiprocessing.cpu_count()):
    ids = [keys[idx] for idx in range(i*step, min(len(ids2refs),(i+1)*step))]
    t = Thread(target=_get_refs, args=(ids2refs, stopwords, word_limit, refs_dict, ids))
    threads.append(t)
    t.start()
  for t in threads: t.join()
  return refs_dict
