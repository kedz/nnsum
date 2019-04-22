class LazyTransform(object):
    def __init__(self, tensor):
        self._tensor = tensor
        self._staged_indexes = []
            
    def stage_indexing(self, dim, index):
        self._staged_indexes.append((dim, index))

    def __call__(self):
        for dim, index in self._staged_indexes:
            self._tensor = self._tensor.index_select(dim, index)
        self._staged_indexes = []
        return self._tensor
    
    def dim(self):
        return self._tensor.dim()
