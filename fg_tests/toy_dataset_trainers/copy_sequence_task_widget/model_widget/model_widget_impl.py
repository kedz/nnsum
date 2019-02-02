import ipywidgets as widgets

from collections import OrderedDict


class ModelWidget(object):
    def __init__(self):

        self._model_params = OrderedDict()
        self._w_models = widgets.Dropdown(options=[], value=None,
                                          description="Models:")
        self._w_models.observe(self._select_callback, names=["value"])
        self._w_model_type = widgets.Dropdown(
            options=["Seq2Seq", "Pointer-Generator"],
            value="Seq2Seq", description="Arch.:")

        self._w_src_unk_rate = widgets.BoundedFloatText(
                value=0., min=0., max=1, description="Src Unk Rate (%)")
        self._w_tgt_unk_rate = widgets.BoundedFloatText(
                value=0., min=0., max=1, description="Tgt Unk Rate (%)")
        self._w_emb_dim = widgets.BoundedIntText(value=100, min=1, max=1024,
                                                 description="Emb. Dim")
        self._w_hid_dim = widgets.BoundedIntText(value=100, min=1, max=1024,
                                                 description="Hidden Dim")
        self._w_rnn_cell = widgets.Dropdown(options=["rnn", "gru", "lstm"],
                                            value="gru", 
                                            description="RNN Cell")
        self._w_layers = widgets.Dropdown(options=[1, 2, 3], value=1, 
                                             description="RNN Layers")
        self._w_attn = widgets.Dropdown(options=["none", "dot"],
                                        value="dot", 
                                        description="Attention")
        self._w_model_name = widgets.Text(value="model-1",
                                          description="Model Name:")

        self._w_create_button = widgets.Button(description="Create Model")
        self._w_create_button.on_click(self.create_button_click)
        self._w_message = widgets.Label()

        self._w_main = widgets.VBox([
            widgets.Label("Models"),
            self._w_models,
            self._w_model_type,
            widgets.HBox([self._w_src_unk_rate, self._w_tgt_unk_rate]),
            widgets.HBox([self._w_emb_dim, self._w_hid_dim]),
            widgets.HBox([self._w_rnn_cell, self._w_layers]),
            widgets.HBox([self._w_attn, self._w_model_name]),
            self._w_create_button,
            self._w_message,
        ])

        self._callbacks = []

    def __call__(self):
        return self._w_main

    @property
    def model_name(self):
        return self._w_model_name.value.strip()

    @model_name.setter
    def model_name(self, new_name):
        self._w_model_name.value = new_name

    @property
    def source_unknown_rate(self):
        return self._w_src_unk_rate.value
    
    @source_unknown_rate.setter
    def source_unknown_rate(self, new_val):
        self._w_src_unk_rate.value = new_val
    
    @property
    def target_unknown_rate(self):
        return self._w_tgt_unk_rate.value

    @target_unknown_rate.setter
    def target_unknown_rate(self, new_val):
        self._w_tgt_unk_rate.value = new_val

    @property
    def embedding_dim(self):
        return self._w_emb_dim.value

    @embedding_dim.setter
    def embedding_dim(self, new_dim):
        self._w_emb_dim.value = new_dim

    @property
    def hidden_dim(self):
        return self._w_hid_dim.value

    @hidden_dim.setter
    def hidden_dim(self, new_dim):
        self._w_hid_dim.value = new_dim

    @property
    def rnn_layers(self):
        return self._w_layers.value

    @rnn_layers.setter
    def rnn_layers(self, new_val):
        self._w_layers.value = new_val

    @property
    def rnn_cell(self):
        return self._w_rnn_cell.value

    @rnn_cell.setter
    def rnn_cell(self, new_val):
        self._w_rnn_cell.value = new_val

    @property
    def attention(self):
        return self._w_attn.value

    @property
    def architecture(self):
        return self._w_model_type.value

    @architecture.setter
    def architecture(self, new_val):
        self._w_model_type.value = new_val

    @attention.setter
    def attention(self, new_val):
        self._w_attn.value = new_val

    def create_button_click(self, button):
        name = self.model_name
        if name in self._model_params or name == "":
            self._w_message.value = (
                "Name must be unique non-empty string."
            )
            return
        else:
            self._w_message.value = ""
 
        params = {
            "name": name, "embedding_dim": self.embedding_dim,
            "source_unknown_rate": self.source_unknown_rate,
            "target_unknown_rate": self.target_unknown_rate,
            "hidden_dim": self.hidden_dim, "rnn_cell": self.rnn_cell,
            "rnn_layers": self.rnn_layers, "attention": self.attention,
            "architecture": self.architecture,
        }

        self._model_params[name] = params
        self._w_models.options = self._model_params.keys()
        self.model_name = "model-{}".format(len(self._model_params) + 1)

        for callback in self._callbacks:
            callback(self._model_params)

    def _select_callback(self, change):
        name = change["new"]
        model_params = self._model_params[name]
        self.source_unknown_rate = model_params["source_unknown_rate"]
        self.target_unknown_rate = model_params["target_unknown_rate"]
        self.embedding_dim = model_params["embedding_dim"]
        self.hidden_dim = model_params["hidden_dim"]
        self.rnn_cell = model_params["rnn_cell"]
        self.rnn_layers = model_params["rnn_layers"]
        self.attention = model_params["attention"]
        self.architecture = model_params["architecture"]

    def register_new_models(self, callback):
        self._callbacks.append(callback)

    def get_params(self):
        return self._model_params
