import unittest

import torch
from nnsum.model.summarization_model import SummarizationModel
from nnsum.module import EmbeddingContext
import nnsum

from collections import namedtuple


class TestSentenceEncoder(unittest.TestCase):

    def test_avg_encoder(self):

        Batch = namedtuple(
            "Batch", ["tokens", "num_sentences", "sentence_lengths"])

        tokens = torch.LongTensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0],
             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        num_sentences = torch.LongTensor([4, 3])
        sentence_lengths = torch.LongTensor([[2, 4, 3, 3], [8, 2, 6, 0]])
        batch = Batch(tokens, num_sentences, sentence_lengths)

        vocab = nnsum.io.Vocab.from_word_list(
            [w for w in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvw'])
        emb_ctx = EmbeddingContext(vocab, 5)
        avg_enc = nnsum.module.sentence_encoder.AveragingSentenceEncoder(5)

        model = SummarizationModel(emb_ctx, avg_enc, None)

        documents = model._prepare_input(batch)
        embeddings = model.embeddings(documents)
        expected_enc_emb = embeddings.sum(-2) / torch.FloatTensor(
            [[2, 4, 3, 3], [8, 2, 6, 1]]).unsqueeze(-1)

        enc_emb = model.encode(batch)

        self.assertTrue(torch.all(expected_enc_emb == enc_emb))

    def test_cnn_encoder(self):

        Batch = namedtuple(
            "Batch", ["tokens", "num_sentences", "sentence_lengths"])

        tokens = torch.LongTensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0],
             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        num_sentences = torch.LongTensor([4, 3])
        sentence_lengths = torch.LongTensor([[2, 4, 3, 3], [8, 2, 6, 0]])
        batch = Batch(tokens, num_sentences, sentence_lengths)

        vocab = nnsum.io.Vocab.from_word_list(
            [w for w in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvw'])
        emb_ctx = EmbeddingContext(vocab, 5)
        cnn_enc = nnsum.module.sentence_encoder.CNNSentenceEncoder(5)

        model = SummarizationModel(emb_ctx, cnn_enc, None)

        documents = model._prepare_input(batch)

        emb_1 = model.embeddings(documents[0:1,:num_sentences[0]])
        doc_embs_1 = cnn_enc(emb_1, None)
        emb_2 = model.embeddings(documents[1:2,:num_sentences[1]])
        doc_embs_2 = torch.cat(
            [cnn_enc(emb_2, None),
             torch.FloatTensor([[[0] * cnn_enc.size]])],
            1)
        expected_enc_emb = torch.cat([doc_embs_1, doc_embs_2], 0)

        enc_emb = model.encode(batch)
        
        self.assertTrue(
            torch.all(torch.abs(expected_enc_emb - enc_emb).lt(1e-6)))


    def test_rnn_encoder(self):

        Batch = namedtuple(
            "Batch", ["tokens", "num_sentences", "sentence_lengths"])

        tokens = torch.LongTensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0],
             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        num_sentences = torch.LongTensor([4, 3])
        sentence_lengths = torch.LongTensor([[2, 4, 3, 3], [8, 2, 6, 0]])
        batch = Batch(tokens, num_sentences, sentence_lengths)

        vocab = nnsum.io.Vocab.from_word_list(
            [w for w in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvw'])
        emb_ctx = EmbeddingContext(vocab, 5)
        rnn_enc = nnsum.module.sentence_encoder.RNNSentenceEncoder(5, 5)

        model = SummarizationModel(emb_ctx, rnn_enc, None)

        documents = model._prepare_input(batch)

        expected_enc_emb = []
        for b in range(2):
            sent_embs = []
            for s in range(num_sentences[b]):
                token_indices = documents[b,s,:sentence_lengths[b,s]]
                token_emb = model.embeddings(token_indices.unsqueeze(-1))
                _, sent_emb = rnn_enc.rnn(token_emb)
                sent_embs.append(sent_emb.view(1, -1))
            if len(sent_embs) == 3:
                sent_embs.append(torch.FloatTensor([[0] * rnn_enc.size]))
            sent_embs = torch.cat(sent_embs, 0)
            expected_enc_emb.append(sent_embs)
        expected_enc_emb = torch.cat(expected_enc_emb, 0).view(2, 4, -1)

        enc_emb = model.encode(batch)
        
        self.assertTrue(
            torch.all(torch.abs(expected_enc_emb - enc_emb).lt(1e-6)))

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestSentenceEncoder("test_avg_encoder"))
    suite.addTest(TestSentenceEncoder("test_cnn_encoder"))
    suite.addTest(TestSentenceEncoder("test_rnn_encoder"))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
