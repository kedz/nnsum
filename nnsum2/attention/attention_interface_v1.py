import torch
import torch.nn as nn
from ..module import Module, register_module, hparam_registry
import numpy as np
import nnsum2.layers


@register_module("attention.key_value_query_interface")
class KeyValueQueryInterface(Module):

    hparams = hparam_registry()

    @hparams(default=nnsum2.layers.Identity())
    def key_adaptor(self):
        pass

    @hparams(default=nnsum2.layers.Identity())
    def value_adaptor(self):
        pass

    @hparams(default=nnsum2.layers.Identity())
    def query_adaptor(self):
        pass

    @hparams()
    def kernel(self):
        pass

    def forward(self, key, query, value=None, key_mask=None, query_mask=None,
                state=None): 
        
        if value is None:
            value = key

        key_feats = self.key_adaptor(key)
        query_feats = self.query_adaptor(query)
        value_feats = self.value_adaptor(value)
       
        attention, new_state = self.kernel(
            key_feats, query_feats, key_mask=key_mask, 
            state={"accumulator": state})

        read_values = torch.bmm(attention, value_feats).permute(1, 0, 2)
        attention = attention.permute(1, 0, 2)
        
        if self.kernel.is_stateless:
            accum = None
        else:
            accum = new_state["accumulator"]

        # Returning 
        # attention (query length x batch size x embedding size)
        # accum None or (batch size x 1 x key length)
        # read_values (query length x batch size x values size)
        return attention, accum, read_values

    def initialize_parameters(self):
        self.key_adaptor.initialize_parameters()
        self.value_adaptor.initialize_parameters()
        self.query_adaptor.initialize_parameters()    
        self.kernel.initialize_parameters()    

    def set_dropout(self, dropout):
        self.key_adaptor.set_dropout(dropout)
        self.value_adaptor.set_dropout(dropout)
        self.query_adaptor.set_dropout(dropout)    
        self.kernel.set_dropout(dropout)    
