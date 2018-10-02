from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class Loss(Metric):
    def __init__(self, output_transform=lambda x: x):
        super(Loss, self).__init__(output_transform)

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        sum_loss, num_examples = output        
        self._sum += sum_loss.item()
        self._num_examples += num_examples

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed')
        return self._sum / self._num_examples
