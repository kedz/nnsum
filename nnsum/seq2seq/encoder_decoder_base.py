import torch.nn as nn
from .rnn_state import RNNState
from .greedy_search import GreedySearch


class EncoderDecoderBase(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoderBase, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    def encode(self, batch):
        encoder_output, encoder_state = self.encoder(
            batch["source_input_features"], 
            batch["source_lengths"]) 
        context = {"encoder_output": encoder_output,
                   "source_mask": batch.get("source_mask", None),
                   "source_vocab_map": batch.get("source_vocab_map", None)}
        return RNNState.new_state(encoder_state), context

    def _expand_encoder_multiref(self, encoder_state, context, max_refs):
        steps, batch_size, _ = encoder_state.size()
        encoder_state = encoder_state\
            .unsqueeze(2).repeat(1, 1, max_refs, 1)\
            .view(steps, batch_size * max_refs, -1)

        encoder_output = context["encoder_output"]
        src_steps = context["encoder_output"].size(1)
        context["encoder_output"] = context["encoder_output"]\
            .unsqueeze(1).repeat(1, max_refs, 1, 1)\
            .view(batch_size * max_refs, src_steps, -1)
        
        src_mask = context.get("source_mask", None)
        if src_mask is not None:
            src_mask = src_mask.unsqueeze(1).repeat(1, max_refs, 1)\
                .view(batch_size * max_refs, -1)
            context["source_mask"] = src_mask
        
        src_vmap = context.get("source_vocab_map", None)
        if src_vmap is not None:
            steps = src_vmap.size(1)
            src_vmap = src_vmap.unsqueeze(1).repeat(1, max_refs, 1, 1)\
                .view(batch_size * max_refs, steps, -1) 
            context["source_vocab_map"] = src_vmap

        return encoder_state, context

    def forward(self, batch):
        encoder_state, context = self.encode(batch)

        if "max_references" in batch:
            encoder_state, context = self._expand_encoder_multiref(
                encoder_state, context, batch["max_references"])
        
        decoder_inputs = batch["target_input_features"]["tokens"]
        return self.decoder(encoder_state, decoder_inputs, context)

    def greedy_decode(self, batch, max_steps=300):
        encoder_state, context = self.encode(batch)
        return GreedySearch(self.decoder, encoder_state, context, 
                            max_steps=max_steps)


    def initialize_parameters(self, verbose=False):
        if verbose:
            print("Initializing model params.")
        self.encoder.initialize_parameters()
        self.decoder.initialize_parameters()
