import pytest
import torch
import torch.nn as nn
import nnsum.embedding_context as ec
import nnsum.seq2seq as s2s
from nnsum.data import Seq2SeqDataLoader



class Dataset(object):

    def __init__(self):
        self._data = [
            {"source": {"sequence": {"tokens": ["A", "B", "C"]}},
             "target": {"sequence": {"tokens": ["C", "B", "A"]}}},
            {"source": {"sequence": {"tokens": ["1", "2", "3"]}},
             "target": {"sequence": {"tokens": ["3", "2", "1"]}}},
            {"source": {"sequence": {"tokens": ["X", "Y", "Z"]}},
             "target": {"sequence": {"tokens": ["Z", "Y", "X"]}}},
        ]

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)

class Model(nn.Module):
    def __init__(self, decoder_params, source_vocab, target_vocab):
        super(Model, self).__init__()
        src_emb = ec.EmbeddingContext(
            source_vocab, embedding_size=100, name="tokens")
        self.encoder = s2s.RNNEncoder(
            src_emb, hidden_dim=100, num_layers=1, 
            rnn_cell=decoder_params["rnn_cell"])

        tgt_emb = ec.EmbeddingContext(
            target_vocab, embedding_size=100, name="tokens")
        self.decoder = s2s.RNNDecoder(
            tgt_emb, hidden_dim=100, num_layers=1, 
            rnn_cell=decoder_params["rnn_cell"],
            attention=decoder_params["attention"])

    def initialize_parameters(self):
        self.encoder.initialize_parameters()
        self.decoder.initialize_parameters()

    def get_context(self, batch):
        encoder_output, encoder_state = self.encoder(
            batch["source_input_features"], 
            batch["source_lengths"]) 
        context = {"encoder_output": encoder_output,
                   "source_mask": batch.get("source_mask", None),
                   "source_vocab_map": batch.get("source_vocab_map", None)}
        return s2s.RNNState.new_state(encoder_state), context

    def forward(self, batch, compute_log_probability=False):
        encoder_state, context = self.get_context(batch)
        return self.decoder(
            encoder_state, 
            batch["target_input_features"]["tokens"], 
            context)

    def get_state(self, batch, compute_log_probability=False,
                  compute_output=False):
        encoder_state, context = self.get_context(batch)
        input_state = s2s.SearchState(
            output=batch["target_input_features"]["tokens"].t(),
            rnn_state=encoder_state)
        return self.decoder.next_state(
            input_state, context, 
            compute_log_probability=compute_log_probability,
            compute_output=compute_output)

    def encode(self, batch):
        encoder_output, encoder_state = self.encoder(
            batch["source_input_features"]["tokens"], 
            batch["source_lengths"]) 
        context = {"encoder_output": encoder_output,
                   "source_mask": batch.get("source_mask", None),
                   "source_vocab_map": batch.get("source_vocab_map", None)}
        return s2s.RNNState.new_state(encoder_state), context

    def greedy_decode(self, batch, max_steps=300):
        encoder_state, context = self.encode(batch)
        return s2s.GreedySearch(self.decoder, encoder_state, context, 
                                max_steps=max_steps)



def decoder_param_gen():
    return [
     #   {"attention": "none", "rnn_cell": "gru"},
     #   {"attention": "none", "rnn_cell": "lstm"}, 
        {"attention": "dot", "rnn_cell": "gru"},
        {"attention": "dot", "rnn_cell": "lstm"},
    ]

def decoder_param_names(params):
    return ("attn={attention}:rcell={rnn_cell}").format(**params)

@pytest.fixture(scope="module", params=decoder_param_gen(), 
                ids=decoder_param_names)
def decoder_params(request):
    return request.param

@pytest.fixture(scope="module")
def dataset():
    return Dataset()

def train_model(model, dataset):
    model.initialize_parameters()
    model.train()
    batches = Seq2SeqDataLoader(
        dataset, 
        model.encoder.embedding_context.named_vocabs,
        model.decoder.embedding_context.named_vocabs,
        batch_size=1,
        shuffle=True)

    optim = torch.optim.SGD(model.parameters(), lr=0.25)
    loss_func = s2s.CrossEntropyLoss()
        #pad_index=model.decoder.embedding_context.vocab.pad_index)

    losses = []
    attn_losses = []

    for epoch in range(100):
        tot_xents = 0
        tot_attn = 0
        for batch in batches:
            optim.zero_grad()
            state = model(batch)
            attn_probs = 0
            for step, i in enumerate(range(3)):
                t = 3 - i
                attn_probs += torch.log(state["context_attention"][step,0,t])
            attn_loss = -attn_probs / 3
            loss = loss_func(state, batch)
            (loss + attn_loss).backward()
            optim.step()
            tot_xents += loss.item()
            tot_attn += attn_loss.item()
        losses.append(tot_xents / len(batches))
        attn_losses.append(tot_attn / len(batches))
    print(losses[0], losses[-1])
    print(attn_losses[0], attn_losses[-1])

    return model

@pytest.fixture(scope="module")
def no_copy_model(decoder_params, dataset):
    source_vocab = ec.Vocab.from_word_list(["A", "B", "C", "1", "2", "3",
                                            "X", "Y", "Z"], 
                                           start="<sos>", pad="<pad>", 
                                           unk="<unk>")
    target_vocab = ec.Vocab.from_word_list(["A", "B", "C", "1", "2", "3",
                                            "X", "Y", "Z"], 
                                           start="<sos>", stop="<eos>", 
                                           pad="<pad>", unk="<unk>")

    model = Model(decoder_params, source_vocab, target_vocab)
    return train_model(model, dataset)

@pytest.fixture(scope="module")
def copy_model(decoder_params, dataset):
    source_vocab = ec.Vocab.from_word_list(["A", "B", "C", "1", "2", "3",
                                            "X", "Y", "Z"], 
                                           start="<sos>", pad="<pad>", 
                                           unk="<unk>")
    target_vocab = ec.Vocab.from_word_list(["B", "C", "1", "3", "X", "Y",], 
                                           start="<sos>", stop="<eos>", 
                                           pad="<pad>", unk="<unk>")

    model = Model(decoder_params, source_vocab, target_vocab)
    return train_model(model, dataset)

def test_generate_no_copy(no_copy_model, dataset):
    cg = s2s.ConditionalGenerator(no_copy_model)
    for ex in dataset:
        exp_tgt = " ".join(ex["target"]["sequence"]["tokens"])
        act_tgt = cg.generate(ex["source"]["sequence"])
        assert exp_tgt == act_tgt

def test_generate_copy_model_no_replace(copy_model, dataset):
    cg = s2s.ConditionalGenerator(copy_model, replace_unknown=False)
    for i, ex in enumerate(dataset):
        r = len(dataset) - i - 1
        exp_tkns = list(ex["target"]["sequence"]["tokens"])
        exp_tkns[r] = copy_model.decoder.embedding_context.vocab.unknown_token
        exp_tgt = " ".join(exp_tkns)

        act_tgt = cg.generate(ex["source"]["sequence"])
        assert exp_tgt == act_tgt

def test_generate_copy_model_yes_replace(copy_model, dataset):
    cg = s2s.ConditionalGenerator(copy_model)
    for i, ex in enumerate(dataset):
        r = len(dataset) - i - 1
        exp_tgt = " ".join(ex["target"]["sequence"]["tokens"])
        act_tgt = cg.generate(ex["source"]["sequence"])
        assert exp_tgt == act_tgt
