import torch


class NoScheduler(object):
    def __init__(self, optimizer):
        self._optimizer = optimizer

    @property
    def optimizer(self):
        return self._optimizer
    
    @property
    def metric(self):
        return self._metric

    def step(self, validation_metric):
        pass

class DecreaseOnPlateauScheduler(object):
    def __init__(self, optimizer, metric, patience=10, decay_factor=.1):
        self._optimizer = optimizer
        self._metric = metric

        mode = "max" if metric in ["nist", "bleu", "accuracy", "f1"] else "min"
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode, factor=decay_factor, 
            patience=patience, verbose=True)

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def metric(self):
        return self._metric

    def step(self, validation_metric):
        self._scheduler.step(validation_metric)
