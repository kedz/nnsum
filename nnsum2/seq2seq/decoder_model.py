from ..module import Module, register_module, hparam_registry


@register_module("seq2seq.decoder_model")
class DecoderModel(Module):

    hparams = hparam_registry()

    @hparams()
    def decoder(self):
        pass

    def encode(self, inputs):
        context = {"encoder_output": None,
                   "source_mask": None,
                   "source_extended_vocab_map": None,
                   "extended_vocab": None,
                   "controls": inputs.get("controls", None)}
        return context, None

    def forward(self, inputs, encoded_inputs=None):

        if encoded_inputs is None:
            encoded_inputs = self.encode(inputs)

        context, encoder_state = encoded_inputs

        if "max_references" in inputs:
            context, encoder_state = self._expand_encoder_multiref(
                encoder_state, context, inputs["max_references"])

        return self.decoder(encoder_state,
                            inputs["target_input_features"], 
                            context,
                            None)

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

        if context["controls"] is not None:
            raise Exception("Controls for multi-reference not implemented.")

        return context, encoder_state

    def initialize_parameters(self):
        self.decoder.initialize_parameters()

    def set_dropout(self, dropout):
        self.decoder.set_dropout(dropout)
