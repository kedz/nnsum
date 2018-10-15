import unittest

import torch
from nnsum.model.summarization_model import SummarizationModel
from nnsum.module import EmbeddingContext
import nnsum

from collections import namedtuple


class TestSummarizationModel(unittest.TestCase):

    def test_prepare_input(self):

        model = SummarizationModel(None, None, None)
        Batch = namedtuple(
            "Batch", ["tokens", "num_sentences", "sentence_lengths"])

        tokens = torch.LongTensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0],
             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        num_sentences = torch.LongTensor([4, 3])
        sentence_lengths = torch.LongTensor([[2, 4, 3, 3], [8, 2, 6, 0]])
        batch = Batch(tokens, num_sentences, sentence_lengths)
        tokens_batched = model._prepare_input(batch)

        expected_tokens_batched = torch.LongTensor(
            [[[1,   2,  0,  0,  0,  0, 0, 0],
              [3,   4,  5,  6,  0,  0, 0, 0],
              [7,   8,  9,  0,  0,  0, 0, 0],
              [10, 11, 12,  0,  0,  0, 0, 0]],
             [[1,   2,  3,  4,  5,  6, 7, 8],
              [9,  10,  0,  0,  0,  0, 0, 0],
              [11, 12, 13, 14, 15, 16, 0, 0],
              [ 0,  0,  0,  0,  0,  0, 0, 0]]])

        self.assertTrue(torch.all(tokens_batched == expected_tokens_batched))

    def test_sort_sentences(self):

        model = SummarizationModel(None, None, None)
        tokens = torch.LongTensor(
            [[[1,   2,  0,  0,  0,  0, 0, 0],
              [3,   4,  5,  6,  0,  0, 0, 0],
              [7,   8,  9,  0,  0,  0, 0, 0],
              [10, 11, 12,  0,  0,  0, 0, 0]],
             [[1,   2,  3,  4,  5,  6, 7, 8],
              [9,  10,  0,  0,  0,  0, 0, 0],
              [11, 12, 13, 14, 15, 16, 0, 0],
              [ 0,  0,  0,  0,  0,  0, 0, 0]]])

        sentence_lengths = torch.LongTensor([[2, 4, 3, 3], [8, 2, 6, 0]])

        expected_sorted_tokens = torch.LongTensor(
            [[1,   2,  3,  4,  5,  6, 7, 8],
             [11, 12, 13, 14, 15, 16, 0, 0],
             [3,   4,  5,  6,  0,  0, 0, 0],
             [7,   8,  9,  0,  0,  0, 0, 0],
             [10, 11, 12,  0,  0,  0, 0, 0],
             [1,   2,  0,  0,  0,  0, 0, 0],
             [9,  10,  0,  0,  0,  0, 0, 0],
             [ 0,  0,  0,  0,  0,  0, 0, 0]])

        expected_sorted_word_counts = torch.LongTensor(
            [8, 6, 4, 3, 3, 2, 2, 1])

        sorted_tokens, sorted_word_counts, inv_order = model._sort_sentences(
            tokens, sentence_lengths)

        self.assertTrue(torch.all(sorted_tokens == expected_sorted_tokens))
        self.assertTrue(
            torch.all(sorted_word_counts == expected_sorted_word_counts))

        expected_inv_order = torch.LongTensor([5, 2, 3, 4, 0, 6, 1, 7])

        self.assertTrue(torch.all(inv_order == expected_inv_order))

        recovered_tokens = sorted_tokens.view(8, -1)[inv_order].view(2, 4, -1)
        
        self.assertTrue(torch.all(recovered_tokens == tokens))

    def test_sorted_sentence_encoder(self):

        vocab = nnsum.io.Vocab.from_word_list(
            [w for w in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvw'])
        emb_ctx = EmbeddingContext(vocab, 5)
        avg_enc = nnsum.module.sentence_encoder.AveragingSentenceEncoder(5)
        model = SummarizationModel(emb_ctx, avg_enc, None)
        Batch = namedtuple(
            "Batch", ["tokens", "num_sentences", "sentence_lengths"])

        tokens = torch.LongTensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0],
             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        num_sentences = torch.LongTensor([4, 3])
        sentence_lengths = torch.LongTensor([[2, 4, 3, 3], [8, 2, 6, 0]])
        batch = Batch(tokens, num_sentences, sentence_lengths)

        tokens_batched = model._prepare_input(batch)

        sorted_enc_sent = model._sort_and_encode(
            tokens_batched, num_sentences, sentence_lengths)
        
        expected_enc_sent = model._encode(
            tokens_batched, num_sentences, sentence_lengths)

        self.assertTrue(torch.all(sorted_enc_sent == expected_enc_sent))

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestSummarizationModel("test_prepare_input"))
    suite.addTest(TestSummarizationModel("test_sort_sentences"))
    suite.addTest(TestSummarizationModel("test_sorted_sentence_encoder"))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
