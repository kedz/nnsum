import pytest
from nnsum.embedding_context import Vocab
import random


def test_counts():
    wl_in = list("abcdef")
    wl_out = list("ghijklm")
    counts = {w: random.randint(0, 100) for w in wl_in + wl_out}

    vocab = Vocab.from_word_list(wl_in, counts=counts)
    for w in wl_in + wl_out:
        assert counts[w] == vocab.count(w)
    
    assert vocab.count("Z") == 0

    no_count = Vocab.from_word_list(wl_in)
    
    with pytest.raises(Exception) as exc_info:
        no_count.count(wl_in[0])
        assert exc_info.value == "Vocab object has no token count data."

def test_unknown():
    
    unk_token = "__UNK__"
    wl = list("abcdef")
    vocab = Vocab.from_word_list(wl, unk=unk_token)
    assert len(vocab) == len(wl) + 1
    assert vocab["Z"] == vocab.unknown_index
    assert vocab[vocab.unknown_index] == unk_token
    assert vocab[vocab.unknown_index] == vocab.unknown_token
    assert unk_token == vocab.unknown_token
    assert vocab.unknown_index == 0
   
    no_unk_vocab = Vocab.from_word_list(wl)
    msg = "Found unknown token (Z) but no unknown index is set."
    with pytest.raises(Exception) as exc_info: 
        vocab["Z"]
        assert exc_info.value == msg
    assert no_unk_vocab.unknown_index is None
    assert no_unk_vocab.unknown_token is None
    
    wl2 = ["alpha", unk_token, "omega"]
    vocab3 = Vocab.from_word_list(wl2, unk=unknown_token)
    assert len(vocab3) == 3
    assert vocab3[vocab3.unknown_index] == unk_token
    assert vocab3[vocab3.unknown_index] == vocab3.unknown_token
    assert unk_token == vocab3.unknown_token
    assert vocab3.unknown_index == 1

def test_start():
    start_token = "__START__"
    wl1 = list("abcdef")
    vocab1 = Vocab.from_word_list(wl1, start=start_token)
    assert len(vocab1) == len(wl1) + 1
    assert vocab1[start_token] == vocab1.start_index
    assert vocab1[vocab1.start_index] == start_token
    assert vocab1[vocab1.start_index] == vocab1.start_token
    assert start_token == vocab1.start_token
    assert vocab1.start_index == 0
   
    vocab2 = Vocab.from_word_list(wl1)
    assert vocab2.start_index is None
    assert vocab2.start_token is None
    
    wl2 = ["alpha", start_token, "omega"]
    vocab3 = Vocab.from_word_list(wl2, start=start_token)
    assert len(vocab3) == 3
    assert vocab3[vocab3.start_index] == start_token
    assert vocab3[vocab3.start_index] == vocab3.start_token
    assert start_token == vocab3.start_token
    assert vocab3.start_index == 1

def test_stop():
    stop_token = "__STOP__"
    wl1 = list("abcdef")
    vocab1 = Vocab.from_word_list(wl1, stop=stop_token)
    assert len(vocab1) == len(wl1) + 1
    assert vocab1[stop_token] == vocab1.stop_index
    assert vocab1[vocab1.stop_index] == stop_token
    assert vocab1[vocab1.stop_index] == vocab1.stop_token
    assert stop_token == vocab1.stop_token
    assert vocab1.stop_index == 0
   
    vocab2 = Vocab.from_word_list(wl1)
    assert vocab2.stop_index is None
    assert vocab2.stop_token is None
    
    wl2 = ["alpha", stop_token, "omega"]
    vocab3 = Vocab.from_word_list(wl2, stop=stop_token)
    assert len(vocab3) == 3
    assert vocab3[vocab3.stop_index] == stop_token
    assert vocab3[vocab3.stop_index] == vocab3.stop_token
    assert stop_token == vocab3.stop_token
    assert vocab3.stop_index == 1

def test_pad():
    pad_token = "__PAD__"
    wl1 = list("abcdef")
    vocab1 = Vocab.from_word_list(wl1, pad=pad_token)
    assert len(vocab1) == len(wl1) + 1
    assert vocab1[pad_token] == vocab1.pad_index
    assert vocab1[vocab1.pad_index] == pad_token
    assert vocab1[vocab1.pad_index] == vocab1.pad_token
    assert pad_token == vocab1.pad_token
    assert vocab1.pad_index == 0
   
    vocab2 = Vocab.from_word_list(wl1)
    assert vocab2.pad_index is None
    assert vocab2.pad_token is None
    
    wl2 = ["alpha", pad_token, "omega"]
    vocab3 = Vocab.from_word_list(wl2, pad=pad_token)
    assert len(vocab3) == 3
    assert vocab3[vocab3.pad_index] == pad_token
    assert vocab3[vocab3.pad_index] == vocab3.pad_token
    assert pad_token == vocab3.pad_token
    assert vocab3.pad_index == 1

def test_word_list():

    wl = list("abcdef")
    vocab = Vocab.from_word_list(wl)
    
    for i, w in enumerate(wl):
        assert w in vocab
        assert vocab[w] == i
        assert vocab[i] == w
    assert len(wl) == len(vocab)

def test_from_vocab_size():

    vsize = 5
    vocab = Vocab.from_vocab_size(vsize)

    for i in range(vsize):
        w = str(i)
        assert w in vocab
        assert vocab[w] == i
        assert vocab[i] == w
    assert vsize == len(vocab)

def test_unknown():
    wl = list("abcdef")
    unk = "_UNK_"
    vocab = Vocab.from_word_list(wl, unk=unk)

    assert unk in vocab
    assert vocab[unk] == 0
    assert "z" not in vocab
    assert vocab["z"] == vocab[unk]

    no_unk_vocab = Vocab.from_word_list(wl)

    with pytest.raises(Exception):
        no_unk_vocab["z"]
