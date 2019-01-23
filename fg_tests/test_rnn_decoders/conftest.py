import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import nnsum.embedding_context as ec
import nnsum.seq2seq as s2s
from nnsum.util import batch_pad_and_stack_vector
from itertools import product


@pytest.fixture(scope="module")
def target_texts(data_params):
    if data_params["target_texts"] == "simple":
        return ["A1 A2 A3",
                "B1",
                "C1 C2 C3 C4"]
    elif data_params["target_texts"] == "beam":
        return ["A B1 C",
                "A B2 C D",
                "A B3 C4 D5 E5",
                "1 2 3A 4",
                "1 2 3B X Y",
                "1 2 3C I J K"]
        return ["A B1 C"] * 3 + \
               ["A B2 C D"] * 2 + \
               ["A B3 C4 D5 E5"] * 1 + \
               ["1 2 3A 4"] * 3 + \
               ["1 2 3B X Y"] * 2 + \
               ["1 2 3C I J K"] * 1 

    else:
        raise Exception()

@pytest.fixture(scope="module")
def source_texts(data_params, decoder_params):
    if data_params["target_texts"] == "simple":
        return ["A2 A3 1 2 3",
                "X Y Z A Z",
                "C3 2 3 4 C2"]
    else:
        return None


@pytest.fixture(scope="module")
def vocab(data_params, target_texts):
    return make_vocab(target_texts, start="<START>", stop="<STOP>", 
                      pad="<PAD>", unk="<UNK>")

@pytest.fixture(scope="module")
def train_data(data_params, decoder_params, vocab, target_texts):
    #if decoder_params["copy_attention"] == "none":
    return make_dataset(vocab, target_texts)
    #else:
    #    return make_dataset(vocab, target_texts, src_texts=source_texts)

@pytest.fixture(scope="module")
def eval_data(data_params, vocab, target_texts):
    return make_dataset(vocab, target_texts)

@pytest.fixture(scope="module")
def batch_size(data_params, target_texts):
    if data_params["target_texts"] == "simple":
        return len(target_texts)
    elif data_params["target_texts"] == "beam":
        return 2
    else:
        raise Exception()

@pytest.fixture(scope="module")
def beam_size(data_params):
    if data_params["target_texts"] == "simple":
        return None
    elif data_params["target_texts"] == "beam":
        return 3
    else:
        raise Exception()

@pytest.fixture(scope="module")
def encoder_state(decoder_params, batch_size):
    if decoder_params["rnn_cell"] == "lstm":
        return s2s.RNNState(
            nn.Parameter(torch.FloatTensor(1, batch_size, 50).normal_()),
            nn.Parameter(torch.FloatTensor(1, batch_size, 50).normal_()))
    else:
        return nn.Parameter(torch.FloatTensor(1, batch_size, 50).normal_())

@pytest.fixture(scope="module")
def context(batch_size):
    return nn.Parameter(torch.FloatTensor(batch_size, 5, 50).normal_())

@pytest.fixture(scope="module")
def trainable_parameters(decoder_params):
    def parameters(decoder, encoder_state, context):
        params = list(decoder.parameters())
        if decoder_params["rnn_cell"] == "lstm":
            params.extend([encoder_state[0], encoder_state[1]])
        else:
            params.append(encoder_state)
        if decoder_params["attention"] != "none":
            params.append(context)
        return params
    return parameters

@pytest.fixture(scope="module")
def named_trainable_parameters(decoder_params):
    def parameters(decoder, encoder_state, context):
        params = list(decoder.named_parameters())
        if decoder_params["rnn_cell"] == "lstm":
            params.extend([("encoder_state0", encoder_state[0]), 
                           ("encoder_state1", encoder_state[1])])
        else:
            params.append(("encoder_state", encoder_state))
        if decoder_params["attention"] != "none":
            params.append(("context", context))
        return params
    return parameters

@pytest.fixture(scope="module")
def initialize_decoder(decoder_params, batch_size, beam_size):
    if beam_size is None:
        def initializer(encoder_state, context):
            return encoder_state, context
    else:
        def initializer(encoder_state, context):
            encoder_state = encoder_state.repeat(1, 1, beam_size).view(
                1, batch_size * beam_size, -1)
            if context is not None:
                ctx_steps = context.size(1)
                context = context.repeat(1, beam_size, 1).view(
                    batch_size * beam_size, ctx_steps, -1)
            return encoder_state, context
    return initializer

@pytest.fixture(scope="module")
def decoder(decoder_params, vocab, train_data, encoder_state, context, 
            trainable_parameters, initialize_decoder):

    max_steps = 150
    emb = ec.EmbeddingContext(vocab, embedding_size=50, name="tokens")
    dec = s2s.RNNDecoder(emb, hidden_dim=50, num_layers=1, 
                         rnn_cell=decoder_params["rnn_cell"],
                         attention=decoder_params["attention"])
                         #copy_attention=decoder_params["copy_attention"])
    dec.initialize_parameters()
    dec.train()
    #optim = getattr(torch.optim, optim)(param_iter_func(), lr=lr)
    optim = torch.optim.SGD(
        trainable_parameters(dec, encoder_state, context), lr=.75)
    losses = []
    for step in range(max_steps):
        optim.zero_grad()
        istate, ictx = initialize_decoder(encoder_state, context)
        state = dec.next_state(
            istate, context=ictx, 
            inputs=train_data["target_input_features"])
 #           target_context_features=train_data["target_context_features"])

        total_xent = F.cross_entropy(
            state["logits"].permute(0, 2, 1), 
            train_data["target_output_features"]["tokens"].t(),
            ignore_index=dec.embedding_context.vocab.pad_index,
            reduction="sum")
        total_tokens = train_data["target_lengths"].sum().float()
        avg_xent = total_xent / total_tokens
        avg_xent.backward()
        optim.step()
        losses.append(avg_xent.item())

    print("Optimized for {} steps. t0={:5.3f} down to t{}={:5.3f}.".format(
            max_steps, losses[0], max_steps, losses[-1]))
    dec.eval()
    return dec

def make_vocab(texts, start=None, stop=None, pad=None, unk=None):
    words = set()
    for text_or_list in texts:
        if isinstance(text_or_list, str):
            text = text_or_list
            for word in text.split():
                words.add(word)
        else:
            for text in text_or_list:
                for word in text.split():
                    words.add(word)
    vocab = ec.Vocab.from_word_list(
        list(words), start=start, stop=stop, pad=pad, unk=unk)
    return vocab

def make_dataset(tgt_vcb, tgt_texts, src_texts=None):
    tgt_lengths = []
    tgt_inp = []
    tgt_out = []
    src_attn = []
    for i, ref in enumerate(tgt_texts):
        ref_toks = [tgt_vcb[t] for t in ref.split()]
        tgt_lengths.append(len(ref_toks) + 1)
        tgt_inp.append(torch.LongTensor([tgt_vcb.start_index] + ref_toks))
        tgt_out.append(torch.LongTensor(ref_toks + [tgt_vcb.stop_index]))
    tgt_lengths = torch.LongTensor(tgt_lengths)
    tgt_inp = batch_pad_and_stack_vector(tgt_inp, tgt_vcb.pad_index)
    tgt_out = batch_pad_and_stack_vector(tgt_out, tgt_vcb.pad_index)

    batch = {
        "target_input_features": {"tokens": tgt_inp},
        "target_output_features": {"tokens": tgt_out},
        "target_lengths": tgt_lengths,
    }

    return batch

#?    if src_texts:
#?        
#?        src_vcb = make_vocab(src_texts)
#?
#?        print()
#?        src_feats = []
#?        for i, ctx in enumerate(src_texts):
#?            print(ctx)
#?            src_feats.append(
#?                torch.LongTensor([src_vcb[t] for t in ctx.split()]))
#?        src_feats = batch_pad_and_stack_vector(src_feats, tgt_vcb.pad_index)
#?
#?        print(src_feats)
#?        print()
#?        context_vocab_map = torch.FloatTensor(src_feats.size(0),
#?                                              src_feats.size(1),
#?                                              len(src_vcb)).zero_()
#?        for b in range(src_feats.size(0)):
#?            print(b)
#?            for i in range(src_feats.size(1)):
#?                context_vocab_map[b,i,src_feats[b,i]] = 1.
#?
#?        print(context_vocab_map)
#?        alignments = torch.LongTensor(tgt_out.size()).zero_()
#?
#?        for i, row in enumerate(tgt_texts):
#?            for j, tok in enumerate(row.split()):
#?
#?
#?            for i, ctx in range(tgt_out.size(1)):
#?                print(tgt_out[b, i])
#?
#?
#?        input()
#?        
#?
#?        ptr_switch = src_feats.ne(tgt_vcb.unknown_index) \
#?                & src_feats.ne(tgt_vcb.pad_index) 
#?        batch["target_context_features"] = {"tokens": src_feats}
#?        batch["pointer_switch"] = ptr_switch
#?
#?    return batch

@pytest.fixture(scope="package")
def tensor_equal():
    def tensor_equal_func(a, b, atol=1e-5):
        if a.dtype == torch.int64:
            return torch.all(a == b)
        else:
            return torch.allclose(a, b, atol=atol)
    return tensor_equal_func

@pytest.fixture(scope="package")
def check_gradient():
    def check_gradient_func(backprop_config, field, params, 
                            zero_gradient_params=None):
        if zero_gradient_params is None:
            zero_gradient_params = []
        output = backprop_config["search"].get_result(field)
        for dims in product(*[range(i) for i in output.size()]):
            backprop_config["optim"].zero_grad()
            for (name, param) in params + zero_gradient_params:
                assert torch.all(param.grad.eq(0.))
            output[dims].backward(retain_graph=True)
            for (name, param) in params:
                assert torch.any(param.grad.ne(0.))
            for (name, param) in zero_gradient_params:
                assert torch.all(param.grad.eq(0.))
    return check_gradient_func
