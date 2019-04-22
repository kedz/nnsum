from ..parameterized import Parameterized
from ..hparam_registry import HParams

import torch
import numpy as np


@Parameterized.register_object("metrics.cloze_ranking")
class ClozeRanking(Parameterized):
    
    hparams = HParams()

    @hparams(default="cloze_score")
    def score_field(self):
        pass

    @hparams(default="cloze_targets")
    def target_field(self):
        pass

    def init_object(self):
        self._sum_ranks = []
        self._num_items = 0
    
    def init_network(self):
        self.reset()
 
    def reset(self):
        self._sum_ranks = []
        self._num_items = 0
        self._cache = None

    def __call__(self, batch, forward_state):

        targets = batch[self.target_field].t().contiguous().view(-1, 1)
        num_items = targets.size(0)
        scores = forward_state[self.score_field].view(num_items, -1)
        
        sort_score, argsort = torch.sort(scores, 1, descending=True)
        E = argsort.eq(targets)
        ranks = np.where(E.cpu().numpy() == 1)[1]
        active_items = targets.view(-1).ne(0).long()
        ranks = ranks * active_items.cpu().numpy()
        self._sum_ranks.append(ranks.sum()) 
        self._num_items += active_items.sum().item()
     
    def compute(self):
        
        results = {"avg_cloze_ranking": sum(self._sum_ranks) / self._num_items}
        self._cache = results
        return results

    def pretty_print(self):
        if self._cache is None:
            results = self.compute()
        else:
            results = self._cache
        print(results)    
