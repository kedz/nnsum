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

    def _prepare_input(self, inputs):
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
 
        return tokens

    def _sort_sentences(self, og_input, og_wc):

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

    def _sort_and_encode(self, documents, document_lengths, sentence_lengths):

        batch_size, doc_size, sent_size = documents.size()

        documents_srt, sentence_lengths_srt, inv_order = self._sort_sentences(
            documents, sentence_lengths) 

        token_embeddings_srt = self.embeddings(documents_srt)

        sentence_embeddings_srt = self.sentence_encoder(
            token_embeddings_srt, sentence_lengths_srt)

        sentence_embeddings = sentence_embeddings_srt[inv_order].view(
            batch_size, doc_size, -1)

        return sentence_embeddings

    def _encode(self, documents, document_lengths, sentence_lengths):
       
        token_embeddings = self.embeddings(documents)

        sentence_embeddings = self.sentence_encoder(
            token_embeddings, sentence_lengths)

        return sentence_embeddings

    def encode(self, input, mask=None):

        if self.sentence_encoder.needs_sorted_sentences:
            encoded_document = self._sort_and_encode(
                input.document, input.num_sentences, input.sentence_lengths)
        else:
            encoded_document = self._encode(
                input.document, input.num_sentences, input.sentence_lengths)

        if mask is not None:
            encoded_document.data.masked_fill_(mask.unsqueeze(2), 0)

        return encoded_document

    def forward(self, input, decoder_supervision=None, mask_logits=False,
                return_attention=False):

        if not self.embeddings.vocab.pad_index is None:
            mask = input.document[:,:,0].eq(self.embeddings.vocab.pad_index)
        else:
            mask = None

        encoded_document = self.encode(input, mask=mask)

        logits_and_attention = self.sentence_extractor(
            encoded_document,
            input.num_sentences, 
            targets=decoder_supervision)

        if isinstance(logits_and_attention, (list, tuple)):
            logits, attention = logits_and_attention
        else:
            logits = logits_and_attention
            attention = None

        if mask_logits and mask is not None:
            logits.data.masked_fill_(mask, float("-inf"))

        if return_attention:
            return logits, attention
        else:
            return logits 

    def predict(self, input, return_indices=False, max_length=100):
        logits = self.forward(input, mask_logits=True)
        batch_size = logits.size(0)
        _, indices = torch.sort(logits, 1, descending=True)

        all_pos = []
        all_text = []
        for b in range(batch_size):
            wc = 0
            text = []
            pos = [] 
            for i in indices.data[b]:
                if i >= input.num_sentences.data[b]:
                    break
                text.append(input.sentence_texts[b][i])
                pos.append(int(i))
                wc += input.pretty_sentence_lengths[b][i]

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

    def token_gradient_magnitude(self, inputs, return_logits=False):

        tokens = self.prepare_input_(inputs)
        batch_size, doc_size, sent_size = tokens.size()

        if self.sentence_encoder.needs_sorted_sentences:
            print("ERROR: Need to debug this first.")
            exit()
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
            token_embeddings.requires_grad = True
            token_embeddings.retain_grad()
            sentence_embeddings = self.sentence_encoder(
                token_embeddings, inputs.sentence_lengths, inputs)

        logits_and_attention = self.sentence_extractor(
            sentence_embeddings,
            inputs.num_sentences, 
            targets=None)

        if isinstance(logits_and_attention, (list, tuple)):
            logits, attention = logits_and_attention
        else:
            logits = logits_and_attention
            attention = None

        mask = tokens.data[:,:,0].eq(0)
        logits.data.masked_fill_(mask, float("-inf"))
        positive_labels = logits.data.new().new(logits.data.shape).fill_(1.) 

        bce = F.binary_cross_entropy_with_logits(
            logits, positive_labels,
            weight=mask.eq(0).float(), 
            reduction='mean')
        bce.backward()

        grad_norms = token_embeddings.grad.norm(p=2, dim=3).data.cpu().numpy()
        
        if return_logits:
            return grad_norms, logits

        else:
            return grad_norms
