import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SummarizationModel(nn.Module):
    def __init__(self, embedding_layer, sentence_encoder, document_encoder,
                 document_decoder):
        super(SummarizationModel, self).__init__()

        self.embeddings = embedding_layer
        self.sentence_encoder = sentence_encoder
        self.document_encoder = document_encoder
        self.document_decoder = document_decoder

    def prepare_input_(self, inputs):
        batch_size = inputs.tokens.size(0)
        sent_size = inputs.sentence_counts.data.max()
        word_size = inputs.sentence_lengths.data.max()

        tokens = inputs.tokens.data.new(batch_size, sent_size, word_size)
        tokens.fill_(0)
        for b in range(batch_size):
            start = 0
            for s in range(inputs.sentence_counts.data[b]):
                length = inputs.sentence_lengths.data[b,s]
                stop = start + length
                tokens[b, s, :length].copy_(inputs.tokens.data[b,start:stop])
                start += length
 
        return Variable(tokens)
    

#    def sort_sentences_(self, original_input, original_word_counts):
#        print(original_input)
#        print(original_word_counts)
#        
#        bs = original_input.size(0)
#        ds = original_input.size(1)
#        ss = original_input.size(2)
#
#        og_input_flat = original_input.view(bs * ds, ss)
#        og_wc_flat = original_word_counts.view(-1)
#
#        srt_wc_flat, argsrt_wc_flat = torch.sort(
#            og_wc_flat, descending=True)
#
#
#        exit()
#
#        sorted_wc, argsort_wc = torch.sort(
#            original_word_counts, 1, descending=True))
#        print(sorted_wc)
#        print(argsort_wc)
#        exit()

    def forward(self, inputs, decoder_supervision=None, mask_logits=False):

        batch_sentences_tokens = self.prepare_input_(inputs)

        if self.sentence_encoder.needs_sorted_sentences:
            self.sort_sentences_(batch_sentences_tokens, inputs.word_count) 


        batch_sentence_token_embeddings = self.embeddings(
            batch_sentences_tokens)

        batch_sentence_embeddings = self.sentence_encoder(
            batch_sentence_token_embeddings, inputs)

        encoder_output, encoder_state = self.document_encoder(
            batch_sentence_embeddings, inputs.sentence_counts)

        logits = self.document_decoder(
            batch_sentence_embeddings, inputs.sentence_counts, 
            encoder_output, encoder_state,
            targets=decoder_supervision)

        if mask_logits:
            mask = batch_sentences_tokens.data[:,:,0].eq(0)
            logits.data.masked_fill_(mask, float("-inf"))

        return logits 

    def predict(self, inputs, metadata, return_indices=False, 
                decoder_supervision=None, max_length=100):
        logits = self.forward(inputs, decoder_supervision=decoder_supervision,
                              mask_logits=True)
        batch_size = logits.size(0)
        _, indices = torch.sort(logits, 1, descending=True)

        all_pos = []
        all_text = []
        for b in range(batch_size):
            wc = 0
            text = []
            pos = [] 
            for i in indices.data[b]:
                if i >= inputs.sentence_counts.data[b]:
                    break
                text.append(metadata.text[b][i])
                pos.append(i)
                wc += inputs.word_count.data[b,i]

                if wc > max_length:
                    break
            all_pos.append(pos)
            all_text.append(text)

        if return_indices:
            return all_text, all_pos
        else:
            return all_text
