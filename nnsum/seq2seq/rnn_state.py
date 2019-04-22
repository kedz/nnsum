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

    def __len__(self):
        return len(self._tensors)

    def __getitem__(self, index):
        return self._tensors[index]

    def __iter__(self):
        for tensor in self._tensors:
            yield tensor

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

    def masked_fill(self, *args, **kwargs):
        return RNNState(*[t.masked_fill(*args, **kwargs)
                          for t in self._tensors])

    def narrow(self, *args, **kwargs):        
        return RNNState(*[t.narrow(*args, **kwargs)
                          for t in self._tensors])

    def squeeze(self, *args, **kwargs):
        return RNNState(*[t.squeeze(*args, **kwargs)
                          for t in self._tensors])

    @property
    def grad(self):
        return RNNState(*[t.grad for t in self._tensors])

    def clone(self, *args, **kwargs):
        return RNNState(*[t.clone(*args, **kwargs) for t in self._tensors])
        
    def fill_(self, *args, **kwargs):
        for t in self._tensors:
            t.fill_(*args, **kwargs)
        return self

    def index_select(self, *args, **kwargs):
        return RNNState(*[t.index_select(*args, **kwargs)
                          for t in self._tensors])
