import torch

import nnsum2.torch as ntorch
from .lazy_transform import LazyTransform


class SearchState(object):

    @staticmethod
    def consolidate(search_states):
        cons_state = search_states[0]
        for state in search_states[1:]:
            cons_state.append(state)
        return cons_state

    def __init__(self, **kwargs):
        self._state = {}
        self._dim_names = {}
        #self._staged_indices = {}
        for field, data in kwargs.items():
            if data is None:
                self._state[field] = None
                self._dim_names[field] = None
            else:
                if data[0].dim() != len(data[1]):
                    raise Exception(
                        "Field {} with dims {} was given dim names: {}".format(
                            field, data[0].dim(), data[1]))
                self._state[field] = LazyTransform(data[0])
                self._dim_names[field] = data[1]
                #self._staged_indices[field] = []

    def __getitem__(self, item):
        val = self._state[item]
        if val is None:
            return val
        if isinstance(val, list):
            val = self._consolidate(item, val)
        return  val()
        #if len(self._staged_indices.get(item, [])) > 0:
        #    val = self._apply_staged_indices(item)
        #return val

    def __setitem__(self, key, value):
        if key in self._state:
            raise Exception("Cannot set an item twice.")
        self._state[key] = LazyTransform(value[0])
        self._dim_names[key] = value[1]
        #self._staged_indices[key] = []

    def __len__(self):
        return len(self._state)

    def _consolidate(self, key, value):
        dim_names = self._dim_names[key]
        value = [v() for v in value]
        if "sequence" not in dim_names:
            new_value = LazyTransform(ntorch.stack(value))
            new_dim_names = ["sequence"] + list(dim_names)
            self._state[key] = new_value
            self._dim_names[key] = new_dim_names
            return new_value
        else:
            new_value = LazyTransform(
                torch.cat(value, dim=dim_names.index("sequence")))
            self._state[key] = new_value
            return new_value

    def __contains__(self, key):
        return key in self._state

    def append(self, other_state):
        assert isinstance(other_state, SearchState)
        assert len(other_state) == len(self)
        for key in self._state:
            old_val = self._state[key]
            new_val = other_state._state[key]

            old_val_is_list = isinstance(old_val, list)
            new_val_is_list = isinstance(new_val, list)

            if old_val is None and new_val is None:
                self._state[key] = None
            elif old_val_is_list and not new_val_is_list:
                assert old_val[0].dim() == new_val.dim()
                old_val.append(new_val)
            elif not old_val_is_list and not new_val_is_list:
                assert old_val.dim() == new_val.dim()
                self._state[key] = [old_val, new_val]
            elif old_val_is_list and new_val_is_list:
                assert old_val[0].dim() == new_val[0].dim()
                old_val.extend(new_val)
            elif not old_val_is_list and new_val_is_list:
                assert old_val.dim() == new_val[0].dim()
                self._state[key] = [old_val] + new_val
            else:
                raise Exception()
        return self

    def get(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default

    def unsqueeze_dim(self, dim, new_dim_name):
        # TODO This has different behavior than unsqueeze
        new_data = {}
        
        for key in self._state:
            
            val = self[key]
            if val is None:
                new_data[key] = None
                continue

            dim_names = self._dim_names[key]
            if dim not in dim_names:
                raise Exception(
                    "{} does not have dimension {}".format(key, dim))

            dim_index = dim_names.index(dim)
            new_dim_names = dim_names[:dim_index + 1] \
                + tuple([new_dim_name]) \
                + dim_names[dim_index + 1:]
            new_val = val.unsqueeze(dim_index + 1)
            new_data[key] = (new_val, new_dim_names)

        return SearchState(**new_data)

    def repeat_dim(self, dim, size):

        new_data = {}
        
        for key in self._state:
            
            val = self[key]
            if val is None:
                new_data[key] = None
                continue

            dim_names = self._dim_names[key]
            if dim not in dim_names:
                raise Exception(
                    "{} does not have dimension {}".format(key, dim))

            dim_index = dim_names.index(dim)
            
            repeat_sizes = [1] * len(dim_names)
            repeat_sizes.insert(dim_index + 1, size)
            new_sizes = list(val.size())
            new_sizes[dim_index] *= size

            new_val = val.unsqueeze(dim_index + 1).repeat(*repeat_sizes) \
                .view(*new_sizes)

            new_data[key] = (new_val, dim_names)

        return SearchState(**new_data)

    def stage_indexing(self, dim, index):
        for key, val in self._state.items():
            if val is None:
                continue
            val.stage_indexing(self._dim_names[key].index(dim), index)
            #self._staged_indices[key].append((dim, index))

#    def _apply_staged_indices(self, key):
#        val = self._state[key]
#        dim_names = self._dim_names[key]
#        staged_indices = self._staged_indices.get(key, [])
#        while len(staged_indices) > 0:
#            dim_name, staged_index = staged_indices.pop(0)
#            dim_index = dim_names.index(dim_name)
#            val = val.index_select(dim_index, staged_index)
#        self._state[key] = val
#        return val
