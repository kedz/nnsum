from ..module import Module, register_module, hparam_registry


@register_module("seq2seq.encoder_decoder_model")
class EncoderDecoderModel(Module):

    hparams = hparam_registry()

    @hparams()
    def encoder(self):
        pass

    def forward(self, inputs):
        O = self.encoder(inputs["source_input_features"], 
                         inputs["source_lengths"])
        print()
        print("AHHH")
        input()

    def initialize_parameters(self):
        self.encoder.initialize_parameters()
