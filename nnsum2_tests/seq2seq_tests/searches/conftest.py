import pytest
import torch
from nnsum.seq2seq import RNNState
from nnsum.embedding_context import Vocab
import nnsum2
import nnsum2.data.batch_utils as batch_utils
from nnsum2.seq2seq.decoders import RNN, RNNPointerGenerator


@pytest.fixture(scope="module")
def vocab_name():
    return "toks"

@pytest.fixture(scope="module")
def context_dims():
    return 5

@pytest.fixture(scope="function", params=[5], 
                ids=lambda x: "emb_dim={}".format(x))
def embedding_dims(request):
    return request.param

@pytest.fixture(scope="function", params=["rnn"], # , "gru", "lstm"],
                ids=lambda x: x)
def rnn_cell(request):
    return request.param

@pytest.fixture(scope="function", params=[5],
                ids=lambda x: "rnn_dims={}".format(x))
def rnn_dims(request):
    return request.param

@pytest.fixture(scope="function")
def source_data(vocab_name):
    return [
        {"sequence": {vocab_name: ["A", "X", "C", "Y", "Z"]}},
        {"sequence": {vocab_name: ["Q", "R", "B", "D"]}},
        {"sequence": {vocab_name: ["F", "S"]}},
    ]

@pytest.fixture(scope="function")
def extended_vocab(source_data, target_vocab, vocab_name):
    return batch_utils.s2s.extend_vocab(
        source_data, vocab_name, target_vocab)

@pytest.fixture(scope="module")
def target_vocab():
    return Vocab.from_list(list("ABCDEFGHIJKLMNOP"), start="<sos>", 
                           stop="<eos>", unk="<unk>", pad="<pad>")

@pytest.fixture(scope="function")
def input_embedding_context(embedding_dims, target_vocab, vocab_name):
    return nnsum2.embedding_context.EmbeddingContext(
        vocab=target_vocab,
        name=vocab_name,
        embedding_dims=embedding_dims)

@pytest.fixture(scope="function")
def output_embedding_context_factory(target_vocab, vocab_name):
    def out_emb_ctx_builder(dim_size):
        return nnsum2.embedding_context.LabelEmbeddingContext(
            vocab=target_vocab,
            name=vocab_name,
            embedding_dims=dim_size)
    return out_emb_ctx_builder

@pytest.fixture(scope="function")
def rnn(embedding_dims, rnn_dims, rnn_cell):
    return nnsum2.layers.RNN(
        input_dims=embedding_dims,
        rnn_dims=rnn_dims,
        cell=rnn_cell)

@pytest.fixture(scope="function")
def init_rnn_state(rnn):
    if rnn.cell == 'lstm':
        return RNNState.new_state([
            torch.nn.Parameter(
                torch.FloatTensor(rnn.layers, 3, rnn.output_dims).normal_()
            ),
            torch.nn.Parameter(
                torch.FloatTensor(rnn.layers, 3, rnn.output_dims).normal_()
            ),
        ])
    else:
        return torch.nn.Parameter(
            torch.FloatTensor(rnn.layers, 3, rnn.output_dims).normal_())

@pytest.fixture(scope="function")
def attention_mechanism_factory(context_dims):
    params = {
        "none": {
            "__modulename__": 'attention.no_mechanism'
        },
        "bilinear": {
            "__modulename__": 'attention.bilinear_mechanism',
            "accumulate": False,
            "value_dims": context_dims,
        },
        "accum_bilinear": {
            "__modulename__": 'attention.bilinear_mechanism',
            "accumulate": True,
            "value_dims": context_dims,
        },
    }
    def attention_mechanism_builder(config):
        return nnsum2.hparam_parser.HParamParser().parse(params[config])
    return attention_mechanism_builder

@pytest.fixture(scope="function")
def decoder_factory(input_embedding_context, rnn, attention_mechanism_factory,
                    output_embedding_context_factory, context_dims):
    def new_decoder(type="rnn", attention_mechanism="none"):
        attn = attention_mechanism_factory(attention_mechanism)
        out_emb = output_embedding_context_factory(
            rnn.output_dims + attn.output_dims)
        if type == "rnn":
            return RNN(
                input_embedding_context=input_embedding_context,
                rnn=rnn,
                context_attention=attn,
                output_embedding_context=out_emb,
            ).eval()
        elif type == "rnn_pointer_generator":
            return RNNPointerGenerator(
                input_embedding_context=input_embedding_context,
                rnn=rnn,
                context_attention=attn,
                copy_switch=nnsum2.layers.FullyConnected(
                    in_feats=2 * context_dims + rnn.output_dims,
                    out_feats=1,
                    dropout=0.0, 
                    activation="Sigmoid",
                ),
                output_embedding_context=out_emb,
            ).eval()
        else: 
            raise Exception("type must be 'rnn' or 'rnn_pointer_generator'")
    return new_decoder

@pytest.fixture(scope='function')
def context(context_dims, source_data, extended_vocab, vocab_name):

    # Create source token indices under extended vocabulary for pointer 
    # generator models. 
    src_ext_v_map = batch_utils.map_tokens(
        [x["sequence"] for x in source_data], 
        vocab_name, extended_vocab, start_token=True)
    
    context = {
        "encoder_output": torch.nn.Parameter(
            torch.FloatTensor(3, 6, context_dims).normal_()),
        "source_mask": src_ext_v_map.eq(extended_vocab.pad_index),
        "source_extended_vocab_map": src_ext_v_map,
        "extended_vocab": extended_vocab,
    }
    context["source_mask"][1,4:].fill_(1)
    context["source_mask"][2,3:].fill_(1)
    return context
