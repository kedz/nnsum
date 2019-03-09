from ..parameterized import Parameterized
from ..hparam_registry import HParams

import torch


@Parameterized.register_object("optim.sgd")
class SGD(Parameterized):

    hparams = HParams()

    @hparams()
    def lr(self):
        pass
        
    @hparams(default=0)
    def momentum(self):
        pass
        
    @hparams(default=0)
    def dampening(self):
        pass
    
    @hparams(default=0)
    def weight_decay(self):
        pass
        
    @hparams(default=False)
    def nesterov(self):
        pass

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        self._params = new_params
        self._optim = torch.optim.SGD(
            new_params, lr=self.lr, momentum=self.momentum, 
            dampening=self.dampening, weight_decay=self.weight_decay, 
            nesterov=self.nesterov)

    @property
    def optim(self):
        return self._optim

    def init_object(self):
        self._params = None
        self._optim = None

    def zero_grad(self):
        if self._params is None:
            raise Exception("No parameters set!")
        self._optim.zero_grad()

    def step(self):
        if self._params is None:
            raise Exception("No parameters set!")
        self._optim.step()

