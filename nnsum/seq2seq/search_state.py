import torch


class SearchState(object):
    def __init__(self, **kwargs):
        self._state = kwargs

    def __getitem__(self, item):
        val = self._state[item]
        if isinstance(val, list):
            val = self._consolidate(item, val)
        return val

    def __setitem__(self, key, value):
        if key in self._state:
            raise Exception("Cannot set an item twice.")
        self._state[key] = value

    def __len__(self):
        return len(self._state)

    def _consolidate(self, key, value):
        assert value[0].dim() >= 2
        new_value = torch.cat(value, dim=0)
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
    
    @property
    def predictor_logits(self):
        return self._predictor_logits

    @property
    def predictor_log_likelihood(self):
        if self._predictor_log_likelihood is None:
            self._predictor_log_likelihood = (
                torch.log_softmax(self.predictor_logits, dim=2)
            )
        return self._predictor_log_likelihood
