import pytest
import torch
import nnsum
import nnsum2.data.batch_utils as batch_utils
import nnsum2




@pytest.fixture(scope="module")
def vocab_name():
    return "toks"

@pytest.fixture(scope="module")
def source_data(vocab_name):
    data = [
        {"sequence": {vocab_name: ["A1", "12", "13", "14"]}},
        {"sequence": {vocab_name: ["21", "22", "23"]}},
        {"sequence": {vocab_name: ["31", "32"]}},
    ]
    return data

@pytest.fixture(scope="module")
def target_data(vocab_name):
    data = [
        {"sequence": {vocab_name: ["A1", "A2", "A3"]}},
        {"sequence": {vocab_name: ["B1"]}},
        {"sequence": {vocab_name: ["C1", "C2", "C3", "C4"]}},
    ]
    return data

@pytest.fixture(scope="module")
def pg_target_data(vocab_name):
    data = [
        {"sequence": {vocab_name: ["A1", "12", "A3"]}},
        {"sequence": {vocab_name: ["23", "B1"]}},
        {"sequence": {vocab_name: ["C1", "???", "32", "C4"]}},
    ]
    return data


@pytest.fixture(scope="module")
def target_vocabs(target_data, vocab_name):
    vocab = nnsum.embedding_context.Vocab.from_list(
        sorted(set([tok for x in target_data
                    for tok in x["sequence"][vocab_name]])),
        start="<sos>", stop="<eos>", pad="<pad>", unk="<unk>")
    named_vocabs = {vocab_name: vocab}
    return named_vocabs

@pytest.fixture(scope="module")
def extended_vocab(source_data, target_vocabs, vocab_name):
    return batch_utils.s2s.extend_vocab(source_data, vocab_name,
                                        target_vocabs[vocab_name])
    
@pytest.fixture(scope="module")
def context_dims():
    return 5

@pytest.fixture(scope="function")
def context_factory(source_data, context_dims, vocab_name, target_vocabs,
                    extended_vocab):
    
    batch_size = len(source_data)
    seq_size = max([len(s["sequence"][vocab_name]) for s in source_data])

    def context_builder(make_mask, is_pg=False):
        context = {
            "encoder_output": torch.nn.Parameter(
                torch.FloatTensor(batch_size, 1 + seq_size, context_dims)\
                    .normal_()
            )
        }

        if make_mask:
            mask = torch.ByteTensor(batch_size, 1 + seq_size).fill_(0)
            for i, s in enumerate(source_data):    
                mask[i,len(s['sequence'][vocab_name]) + 1:].fill_(1)
            context["source_mask"] = mask

        if is_pg:
            context["extended_vocab"] = extended_vocab
            context["source_extended_vocab_map"] = batch_utils.map_tokens(
                [x['sequence'] for x in source_data],
                vocab_name, extended_vocab, start_token=True)

        return context
    return context_builder

@pytest.fixture(scope="module")
def batch_inputs_factory(target_data, pg_target_data, target_vocabs):
    def batch_inputs_builder(input_type):
        if input_type == "simple":
            return batch_utils.s2s.target(target_data, target_vocabs)
        elif input_type == "pg":
            return batch_utils.s2s.target(pg_target_data, target_vocabs)
        else:
            raise Exception("Options are 'simple' or 'pg'")
    return batch_inputs_builder

@pytest.fixture(scope="function", params=[5], 
                ids=lambda x: "emb_dim={}".format(x))
def embedding_dims(request):
    return request.param

@pytest.fixture(scope="function", params=["rnn", "gru", "lstm"],
                ids=lambda x: x)
def rnn_cell(request):
    return request.param

@pytest.fixture(scope="function", params=[5],
                ids=lambda x: "rnn_dims={}".format(x))
def rnn_dims(request):
    return request.param

@pytest.fixture(scope="function")
def rnn(embedding_dims, rnn_dims, rnn_cell):
    return nnsum2.layers.RNN(
        input_dims=embedding_dims,
        rnn_dims=rnn_dims,
        cell=rnn_cell)

@pytest.fixture(scope="function")
def attention_mechanism_builder(context_dims):
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
    def attention_mechanism_factory(config):
        return nnsum2.hparam_parser.HParamParser().parse(params[config])
    return attention_mechanism_factory

@pytest.fixture(scope="function")
def input_embedding_context(embedding_dims, target_vocabs, vocab_name):
    return nnsum2.embedding_context.EmbeddingContext(
        vocab=target_vocabs[vocab_name],
        name=vocab_name,
        embedding_dims=embedding_dims)

@pytest.fixture(scope="function")
def output_embedding_context(rnn, attention_mechanism, target_vocabs,
                             vocab_name):
    return nnsum2.embedding_context.LabelEmbeddingContext(
        vocab=target_vocabs[vocab_name],
        name=vocab_name,
        embedding_dims=rnn.output_dims + attention_mechanism.output_dims)
