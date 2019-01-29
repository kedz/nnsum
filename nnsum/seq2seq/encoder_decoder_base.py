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
            batch["source_input_features"]["tokens"], 
            batch["source_lengths"]) 
        context = {"encoder_output": encoder_output,
                   "source_mask": batch.get("source_mask", None),
                   "source_vocab_map": batch.get("source_vocab_map", None)}
        return RNNState.new_state(encoder_state), context

    def forward(self, batch):
        encoder_state, context = self.encode(batch)
        
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
