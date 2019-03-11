from ..module import Module, register_module, hparam_registry


@register_module("seq2seq.encoder_decoder_model")
class EncoderDecoderModel(Module):

    hparams = hparam_registry()

    @hparams()
    def encoder(self):
        pass

    @hparams()
    def decoder(self):
        pass

    def encode(self, inputs):
        encoder_output, encoder_state = self.encoder(
            inputs["source_input_features"], 
            inputs["source_lengths"])

        context = {"encoder_output": encoder_output,
                   "source_mask": inputs.get("source_mask", None),
                   "source_vocab_map": inputs.get("source_vocab_map", None)}
        return context, encoder_state

    def forward(self, inputs):
        context, encoder_state = self.encode(inputs)

        if "max_references" in inputs:
            raise Exception("Multireference forward not implemented!")

        return self.decoder(encoder_state,
                            inputs["target_input_features"], 
                            context)

    def initialize_parameters(self):
        self.encoder.initialize_parameters()
        self.decoder.initialize_parameters()
