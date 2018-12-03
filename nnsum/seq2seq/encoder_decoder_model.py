import torch
import torch.nn as nn
from .beam_search import BeamSearch
import numpy as np


class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoderModel, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

    @property
    def encoder(self):
        return self._encoder
    
    @property
    def decoder(self):
        return self._decoder

    def forward(self, inputs):
        context, state = self._encoder(
            inputs["source_features"], 
            inputs["source_lengths"]) 

        logits, attn, state = self._decoder(
            inputs["target_input_features"], context, state)
        return logits

    def predict(self, inputs):
        context, state = self._encoder(
            inputs["source_features"], 
            inputs["source_lengths"]) 
        return self._decoder.predict(context, state)

    def predict_tokens(self, inputs):
        predicted_indices = self.predict(inputs)
        return self.decoder.embedding_context.convert_index_tensor(
            predicted_indices)

    def _sort_inputs(self, inputs):
        src_lengths, sorted_order = torch.sort(
            inputs["source_lengths"], descending=True)
        src_features = {feat: values[sorted_order] 
                        for feat, values in inputs["source_features"].items()}
        
        inv_order = torch.sort(sorted_order)[1]
        new_inputs = {"source_lengths": src_lengths, 
                      "source_features": src_features}
        return new_inputs, inv_order

    def decode(self, inputs, sorted=False, return_tokens=True):
        if not sorted:
            inputs, inv_order = self._sort_inputs(inputs)
            context, state = self._encoder(
                inputs["source_features"], 
                inputs["source_lengths"]) 
            output = self._decoder.decode(context, state)
            output = output[inv_order]
        else:
            context, state = self._encoder(
                inputs["source_features"], 
                inputs["source_lengths"]) 
            output = self._decoder.decode(context, state)

        if return_tokens:
            output = self.decoder.embedding_context.convert_index_tensor(
                output)

        return output

    def beam_decode(self, inputs, beam_size=8, sorted=False, 
                    return_tokens=True, max_steps=300):

        if not sorted:
            inputs, inv_order = self._sort_inputs(inputs)
            context, state = self._encoder(
                inputs["source_features"], 
                inputs["source_lengths"]) 
            beam = BeamSearch(self.decoder, state, context, 
                              beam_size=beam_size, max_steps=max_steps)
            beam.search()
            output = beam.candidates[inv_order]
        else:
            context, state = self._encoder(
                inputs["source_features"], 
                inputs["source_lengths"]) 
            beam = BeamSearch(self.decoder, state, context, 
                              beam_size=beam_size, max_steps=max_steps)
            beam.search()
            output = beam.candidates

        if return_tokens:
            output = self.decoder.embedding_context.convert_index_tensor(
                output)

        return output
