import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(BaseModel, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

    @property
    def encoder(self):
        return self._encoder
    
    @property
    def decoder(self):
        return self._decoder

    def forward(self, inputs):
        pass

    def xentropy(self, inputs):
        encoder_context, encoder_state = self.encode(inputs)
        mask = inputs.get("source_mask", None)
        log_likelihood, decoder_state = self.decoder.log_likelihood(
            inputs["target_input_features"], 
            inputs["target_output_features"], 
            encoder_context, 
            encoder_state, 
            context_mask=mask)

    def encode(self, inputs):
        return self.encoder(inputs["source_features"], 
                            inputs["source_lengths"])

    def decode(self):
        pass

    def beam_decode(self):
        pass

    def initialize_parameters(self):
        print(" Initializing encoder parameters.")
        self.encoder.initialize_parameters()
        print(" Initializing decoder parameters.")
        self.decoder.initialize_parameters()
