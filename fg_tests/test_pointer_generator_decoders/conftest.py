import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import nnsum.embedding_context as ec
import nnsum.seq2seq as s2s
from nnsum.data import seq2seq_batcher


@pytest.fixture(scope="module")
def source_texts():
    return [
        "Rufus the cat chased Terry the dog",
        "Regina played the hurdy gurdy",
        "Sally designed the program"
    ]

@pytest.fixture(scope="module")
def target_texts():
    return [
        "Rufus chases Terry",
        "Regina plays the instrument",
        "Sally designs programs"
    ]

@pytest.fixture(scope="module")
def alignments(source_texts, target_texts):
    alignments = []
    for src, tgt in zip(source_texts, target_texts):
        src = src.split()
        alignments_i = []
        for tgt_tok in tgt.split():
            try:
                a = src.index(tgt_tok)
            except ValueError:
                a = -1
            alignments_i.append(a)
        alignments.append(alignments_i)
    return alignments

@pytest.fixture(scope="module")
def batch_size():
    return 3

@pytest.fixture(scope="module")
def source_vocab(source_texts):
    word_list = [token for string in source_texts for token in string.split()]
    return ec.Vocab.from_word_list(word_list, pad="<PAD>", unk="<UNK>",
                                   start="<START>")

@pytest.fixture(scope="module")
def target_vocab():
    word_list = ["chases", "chased", "plays", "played", "designs", "designed",
                 "the", "instrument", "programs"]
    return ec.Vocab.from_word_list(word_list, pad="<PAD>", unk="<UNK>",
                                   start="<START>", stop="<STOP>")

@pytest.fixture(scope="module")
def train_data(source_texts, target_texts, source_vocab, target_vocab,
               alignments):

    batch = {}
    batch.update(seq2seq_batcher.batch_source(source_texts, source_vocab))
    batch.update(seq2seq_batcher.batch_target(target_texts, target_vocab))
    batch.update(seq2seq_batcher.batch_copy_alignments(
        source_texts, target_texts, target_vocab, alignments,
        sparse_map=False))

    return batch

@pytest.fixture(scope="module")
def model(decoder_params, source_vocab, target_vocab, train_data):
    max_steps = 150
 
    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            src_emb = ec.EmbeddingContext(
                source_vocab, embedding_size=100, name="tokens")
            self.encoder = s2s.RNNEncoder(
                src_emb, hidden_dim=100, num_layers=1, 
                rnn_cell=decoder_params["rnn_cell"])

            tgt_emb = ec.EmbeddingContext(
                target_vocab, embedding_size=100, name="tokens")
            self.decoder = s2s.PointerGeneratorDecoder(
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

    model = TestModel()
    model.initialize_parameters()
    model.train()
    ptr_targets = train_data["copy_targets"].t().unsqueeze(-1)
    gen_targets = train_data["target_output_features"]["tokens"]\
        .t().unsqueeze(-1)
    no_gen_mask = gen_targets != ptr_targets
    mask = ptr_targets.eq(model.decoder.embedding_context.vocab.pad_index)

    optim = torch.optim.SGD(model.parameters(), lr=0.25)
    loss_func = s2s.PointerGeneratorCrossEntropyLoss(
        pad_index=model.decoder.embedding_context.vocab.pad_index)
    losses = []
    
    for step in range(max_steps):
        optim.zero_grad()
                
        state = model(train_data)
        avg_xent = loss_func(state, train_data)
        avg_xent.backward()
        optim.step()
        losses.append(avg_xent.item())

    print("Optimized for {} steps. t0={:5.3f} down to t{}={:5.3f}.".format(
            max_steps, losses[0], max_steps, losses[-1]))
    model.eval()
    return model

@pytest.fixture(scope="package")
def tensor_equal():
    def tensor_equal_func(a, b, atol=1e-5):
        if a.dtype == torch.int64:
            return torch.all(a == b)
        else:
            return torch.allclose(a, b, atol=atol)
    return tensor_equal_func
