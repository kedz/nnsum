class Vocab(object):
    def __init__(self, index2tokens, tokens2index, pad=None, unk=None,
                 start=None, stop=None, counts=None):

        # TODO add checks and tests to constructor. I.e. no duplicate tokens
        # or indices. Assert pad,unk, start, and stop are in dict if
        # they are not none. Think about multivocab stop symbols.

        self._index2tokens = index2tokens
        self._tokens2index = tokens2index
        self._pad = pad
        self._unk = unk
        self._start = start
        self._stop = stop
        self._pad_idx = self._tokens2index.get(pad, None)
        self._unk_idx = self._tokens2index.get(unk, None)
        self._start_idx = self._tokens2index.get(start, None)
        self._stop_idx = self._tokens2index.get(stop, None)
        self._counts = counts

    @staticmethod
    def from_vocab_size(vocab_size, **kwargs):
        word_list = [str(x) for x in range(vocab_size)]
        return Vocab.from_word_list(word_list, **kwargs)

    @staticmethod
    def from_counts(counts, at_least=0, top_k=None, **kwargs):

        wc = sorted(counts.items(), key=lambda x: x[0], reverse=True)
        wc.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None and top_k > 0:
            wc = wc[:top_k]
        word_list = [w for w, c in wc if c >= at_least]
        return Vocab.from_word_list(word_list, counts=counts, **kwargs)

    @staticmethod
    def from_word_list(word_list, pad=None, unk=None, 
                       start=None, stop=None, counts=None):

        if stop is not None and stop not in word_list:
            word_list = [stop] + word_list

        if start is not None and start not in word_list:
            word_list = [start] + word_list

        if unk is not None and unk not in word_list:
            word_list = [unk] + word_list
        
        if pad is not None and pad not in word_list:
            word_list = [pad] + word_list

        word2index = {}
        index2word = []
        for word in word_list:
            if word not in word2index:
                word2index[word] = len(word2index)
                index2word.append(word)
        return Vocab(index2word, word2index, pad=pad, unk=unk, 
                     start=start, stop=stop, counts=counts)

    

    def __getitem__(self, word_or_index):
        if isinstance(word_or_index, str):
            return self.index(word_or_index)
        else:
            return self.token(word_or_index)

    def index(self, token):
        index = self._tokens2index.get(token, self._unk_idx)
        if index is None:
            raise Exception(
                "Found unknown token ({}) but no unknown index is set.".format(
                    token))
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
    def start_token(self):
        return self._start

    @property
    def stop_token(self):
        return self._stop

    @property
    def unknown_index(self):
        return self._unk_idx

    @property
    def pad_index(self):
        return self._pad_idx

    @property
    def start_index(self):
        return self._start_idx

    @property
    def stop_index(self):
        return self._stop_idx

    def enumerate(self):
        return enumerate(self._index2tokens)
           
    def __iter__(self):
        for word in self._index2tokens:
            yield word           
            
    def __contains__(self, token):
        return token in self._tokens2index

    def count(self, token_or_index):
        if isinstance(token_or_index, int):
            token = self[token_or_index]
        else:
            token = token_or_index
        if self._counts is not None:
            return self._counts.get(token, 0)
        else:
            raise Exception("Vocab object has no token count data.")
