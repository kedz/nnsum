import torch
import torch.nn as nn
from .base_model import BaseModel


class RNNEncoderDecoderModel(BaseModel):

    def _check_sort(self, source_lengths):
        batch_size = source_lengths.size(0)

        if batch_size == 1:
            return True
        elif batch_size == 2:
            return source_lengths[0] >= source_lengths[1]
        elif batch_size >= 3:
            if torch.any(source_lengths[:-1] < source_lengths[1:]):
                return False
            return True
        else:
            raise ValueError("source_lengths must have positive dimension.")

    def _sort_inputs(self, inputs):
        sorted_inputs = {}
        
        src_lengths, sorted_order = torch.sort(
            inputs["source_lengths"], descending=True)
        src_features = {feat: values[sorted_order] 
                        for feat, values in inputs["source_features"].items()}

        sorted_inputs["source_lengths"] = src_lengths
        sorted_inputs["source_features"] = src_features

        if "target_input_features" in inputs:
            tgt_in_features = {
                feat: values[sorted_order]
                for feat, values in inputs["target_input_features"].items()}
            tgt_out_features = {
                feat: values[sorted_order]
                for feat, values in inputs["target_output_features"].items()}
            target_lengths = inputs["target_lengths"][sorted_order]
            sorted_inputs["target_input_features"] = tgt_in_features
            sorted_inputs["target_output_features"] = tgt_out_features
            sorted_inputs["target_lengths"] = target_lengths

        if inputs.get("multi_ref", False):
            raise Exception("Multi ref not implemented yet.")
        if inputs.get("source_mask", False):
            raise Exception("source mask not implemented yet.")

        inv_order = torch.sort(sorted_order)[1]
        
        return sorted_inputs, inv_order


    def forward(self, inputs):
        pass

    def xentropy(self, inputs):
        in_order = self._check_sort(inputs["source_lengths"])
        if not in_order:
            inputs, inv_order = self._sort_inputs(inputs)
        xentropy = super(RNNEncoderDecoderModel).xentropy(inputs)
            

#    def encode(self, inputs):
#        pass
        

    def decode(self):
        pass

    def beam_decode(self):
        pass

