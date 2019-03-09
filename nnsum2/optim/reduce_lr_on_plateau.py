from ..parameterized import Parameterized
from ..hparam_registry import HParams

import torch


@Parameterized.register_object("optim.reduce_lr_on_plateau")
class ReduceLROnPlateau(Parameterized):
    
    hparams = HParams()

    @hparams(default='min')
    def mode(self):
        pass
    
    @hparams(default=0.1)
    def factor(self):
        pass
    
    @hparams(default=10)
    def patience(self):
        pass
                     
    @hparams(default=False)
    def verbose(self):
        pass
    
    @hparams(default=1e-4)
    def threshold(self):
        pass
    
    @hparams(default='rel')
    def threshold_mode(self):
        pass
                                      
    @hparams(default=0)
    def cooldown(self):
        pass
    
    @hparams(default=0)
    def min_lr(self):
        pass
    
    @hparams(default=1e-8)
    def eps(self):
        pass

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self._optimizer = new_optimizer
        self._lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, mode=self.mode, factor=self.factor, 
            patience=self.patience, verbose=self.verbose, 
            threshold=self.threshold, threshold_mode=self.threshold_mode,
            cooldown=self.cooldown, min_lr=self.min_lr, eps=self.eps)

    def init_object(self):
        self._optimizer = None
        self._lr_sched = None

    def step(self, value):
        if self._lr_sched is None:
            raise Exception("No optimizer is set!")
        self._lr_sched.step(value)
