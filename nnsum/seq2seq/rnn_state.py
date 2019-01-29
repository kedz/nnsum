import torch


#TODO rename to EncoderState
class RNNState(object):

    class StateIndexer(object):
        def __init__(self, tensors):
            self._tensors = tensors

        def __getitem__(self, indices):
            return RNNState(*[t[indices] for t in self._tensors])

    @staticmethod
    def new_state(tensor_or_tuple):
        if isinstance(tensor_or_tuple, (tuple, list)):
            return RNNState(*tensor_or_tuple)
        else:
            return tensor_or_tuple

    def __init__(self, *tensors):
        self._tensors = tensors
        self.reindex = RNNState.StateIndexer(tensors)

    def __getitem__(self, index):
        return self._tensors[index]

    def size(self, dim=None, state_index=0):
        if dim is None:
            return self._tensors[state_index].size()
        else:
            return self._tensors[state_index].size(dim)

    @property
    def device(self):
        return self._tensors[0].device

    def dim(self, state_index=0):
        return self._tensors[state_index].dim()

    def repeat(self, *args):
        return RNNState(*[t.repeat(*args) for t in self._tensors])
    
    def unsqueeze(self, *args):
        return RNNState(*[t.unsqueeze(*args) for t in self._tensors])

    def view(self, *args):
        return RNNState(*[t.view(*args) for t in self._tensors])

    def new(self, *args, **kwargs):
        return self._tensors[0].new(*args, **kwargs)

    def get(self, index):
        return self._tensors[index]
