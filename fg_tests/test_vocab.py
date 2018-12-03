import pytest
from nnsum.embedding_context import Vocab


def test_word_list():

    wl = "abcdef".split()
    vocab = Vocab.from_word_list(wl, pad=None, unk=None)
    
    for i, w in enumerate(wl):
        assert w in vocab
        assert vocab[w] == i
        assert vocab[i] == w
    assert len(wl) == len(vocab)

def test_from_vocab_size():

    vsize = 5
    vocab = Vocab.from_vocab_size(vsize, pad=None, unk=None)

    for i in range(vsize):
        w = str(i)
        assert w in vocab
        assert vocab[w] == i
        assert vocab[i] == w
    assert vsize == len(vocab)

def test_unknown():
    wl = "abcdef".split()
    unk = "_UNK_"
    vocab = Vocab.from_word_list(wl, pad=None, unk=unk)

    assert unk in vocab
    assert vocab[unk] == 0
    assert "z" not in vocab
    assert vocab["z"] == vocab[unk]

    no_unk_vocab = Vocab.from_word_list(wl, pad=None, unk=None)

    with pytest.raises(Exception):
        no_unk_vocab["z"]
