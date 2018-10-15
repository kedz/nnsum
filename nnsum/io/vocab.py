class Vocab(object):
    def __init__(self, index2tokens, tokens2index, pad="_PAD_", unk="_UNK_"):
        self._index2tokens = index2tokens
        self._tokens2index = tokens2index
        self._pad = pad
        self._unk = unk
        self._pad_idx = self._tokens2index.get(pad, None)
        self._unk_idx = self._tokens2index.get(unk, None)

    @staticmethod
    def from_word_list(word_list, pad="_PAD_", unk="_UNK_"):

        if not unk is None:
            word_list = [unk] + word_list
        
        if not pad is None:
            word_list = [pad] + word_list

        word2index = {w: i for i, w in enumerate(word_list)}
        return Vocab(word_list, word2index, pad=pad, unk=unk)

    def __getitem__(self, word_or_index):
        if isinstance(word_or_index, str):
            return self.index(word_or_index)
        else:
            return self.token(word_or_index)

    def index(self, token):
        index = self._tokens2index.get(token, self._unk_idx)
        if index is None:
            raise Exception(
                "Found unknown token: {} but no unknown index set".format(token))
        else:
            return index

    def token(self, index):
        return self._index2tokens[index]

    def __len__(self):
        return len(self._tokens2index)

    @property
    def unknown_token(self):
        return self._unk

    @property
    def pad_token(self):
        return self._pad

    @property
    def unknown_index(self):
        return self._unk_idx

    @property
    def pad_index(self):
        return self._pad_idx

    def enumerate(self):
        return enumerate(self._index2tokens)
            
    def __contains__(self, token):
        return token in self._tokens2index
