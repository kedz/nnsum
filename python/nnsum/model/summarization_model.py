import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SummarizationModel(nn.Module):
    def __init__(self, embedding_layer, sentence_encoder, sentence_extractor):
        super(SummarizationModel, self).__init__()

        self.embeddings = embedding_layer
        self.sentence_encoder = sentence_encoder
        self.sentence_extractor = sentence_extractor

    def prepare_input_(self, inputs):
        batch_size = inputs.tokens.size(0)
        sent_size = inputs.num_sentences.data.max()
        word_size = inputs.sentence_lengths.data.max()

        tokens = inputs.tokens.data.new(batch_size, sent_size, word_size)
        tokens.fill_(0)
        for b in range(batch_size):
            start = 0
            for s in range(inputs.num_sentences.data[b]):
                length = inputs.sentence_lengths.data[b,s]
                stop = start + length
                tokens[b, s, :length].copy_(inputs.tokens.data[b,start:stop])
                start += length
 
        return Variable(tokens)

    def sort_sentences_(self, og_input, og_wc):

        bs = og_input.size(0)
        ds = og_input.size(1)
        ss = og_input.size(2)
        
        og_input_flat = og_input.contiguous().view(bs * ds, ss)
        og_wc_flat = og_wc.contiguous().view(-1)
        
        srt_wc_flat, argsrt_wc_flat = torch.sort(og_wc_flat, descending=True)
        
        srt_input_flat = og_input_flat[argsrt_wc_flat]
        
        _, inv_order = torch.sort(argsrt_wc_flat)

        srt_wc = Variable(
            srt_wc_flat.data.masked_fill_(srt_wc_flat.data.eq(0), 1))
        srt_inp = Variable(srt_input_flat.data)
        inv_order = Variable(inv_order.data)
        return srt_inp, srt_wc, inv_order

    def forward(self, inputs, decoder_supervision=None, mask_logits=False,
                return_attention=False):

        tokens = self.prepare_input_(inputs)
        batch_size, doc_size, sent_size = tokens.size()

        if self.sentence_encoder.needs_sorted_sentences:
            tokens_srt, word_count_srt, inv_order = self.sort_sentences_(
                tokens, inputs.word_count) 
            token_embeddings_srt = self.embeddings(tokens_srt)
            sentence_embeddings_srt = self.sentence_encoder(
                token_embeddings_srt, word_count_srt, inputs)

            sentence_embeddings_flat = sentence_embeddings_srt[inv_order]
            sentence_embeddings = sentence_embeddings_flat.view(
                batch_size, doc_size, -1)

            mask = tokens.data[:,:,:1].eq(0).repeat(
                1, 1, sentence_embeddings.size(2))
            sentence_embeddings.data.masked_fill_(mask, 0)

        else:
            token_embeddings = self.embeddings(tokens)
            sentence_embeddings = self.sentence_encoder(
                token_embeddings, inputs.sentence_lengths, inputs)

        logits, attention = self.sentence_extractor(
            sentence_embeddings,
            inputs.num_sentences, 
            targets=decoder_supervision)

        if mask_logits:
            mask = tokens.data[:,:,0].eq(0)
            logits.data.masked_fill_(mask, float("-inf"))

        if return_attention:
            return logits, attention
        else:
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
                if i >= inputs.num_sentences.data[b]:
                    break
                text.append(metadata.texts[b][i])
                pos.append(i)
                wc += metadata.sentence_lengths[b][i]

                if wc > max_length:
                    break
            all_pos.append(pos)
            all_text.append(text)

        if return_indices:
            return all_text, all_pos
        else:
            return all_text

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" Model parameter initialization started.")
        for module in self.children():
            module.initialize_parameters(logger=logger)
        if logger:
            logger.info(" Model parameter initialization finished.\n")
