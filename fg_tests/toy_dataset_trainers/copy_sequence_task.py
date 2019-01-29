import argparse
from collections import OrderedDict

import ipywidgets as widgets
import matplotlib.pyplot as plt

import torch

from nnsum.data import Seq2SeqDataLoader
from nnsum.datasets import CopyDataset
import nnsum.embedding_context as ec
import nnsum.seq2seq as s2s


def main():
    parser = argparse.ArgumentParser(
        "Train a seq2seq model to perform a copying task.")
    parser.add_argument("--epoch-size", default=2048, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-epochs", default=25, type=int)
    parser.add_argument("--dataset-vocab-size", default=25, type=int)
    parser.add_argument("--max-length", default=15, type=int)
    parser.add_argument("--seed", default=92224234, type=int)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    test_seed = torch.LongTensor(1).random_(0,2**31).item()
    train_seeds = torch.LongTensor(args.num_epochs).random_(0,2**31).tolist()

    train_dataset = CopyDataset(args.dataset_vocab_size, args.max_length, 
                                args.epoch_size, random_seed=train_seeds[0])

    test_dataset = CopyDataset(args.dataset_vocab_size, args.max_length, 
                               100, random_seed=test_seed)

    src_vocab = ec.Vocab.from_word_list(train_dataset.word_list(), pad="<PAD>",
                                        unk="<UNK>", start="<START>")
    src_vocabs = {"tokens": src_vocab}
    
    tgt_vocab = ec.Vocab.from_word_list(train_dataset.word_list(), pad="<PAD>",
                                        unk="<UNK>", start="<START>", 
                                        stop="<STOP>")
    tgt_vocabs = {"tokens": tgt_vocab}

    train_dataloader = Seq2SeqDataLoader(train_dataset, src_vocabs, tgt_vocabs,
                                         batch_size=args.batch_size,
                                         include_original_data=True)
    test_dataloader = Seq2SeqDataLoader(test_dataset, src_vocabs, tgt_vocabs,
                                        batch_size=2,
                                        include_original_data=True)

    src_emb_ctx = ec.EmbeddingContext(src_vocab, embedding_size=300, 
                                      name="tokens")
    enc = s2s.RNNEncoder(src_emb_ctx, hidden_dim=300, num_layers=1,
                         rnn_cell="lstm")

    tgt_emb_ctx = ec.EmbeddingContext(tgt_vocab, embedding_size=300, 
                                      name="tokens")
    dec = s2s.RNNDecoder(tgt_emb_ctx, hidden_dim=300, num_layers=1,
                         attention="none", rnn_cell="lstm")

    model = s2s.EncoderDecoderBase(enc, dec)
    model.initialize_parameters()

    loss_func = s2s.CrossEntropyLoss(tgt_vocab.pad_index)

    #model.initialize_parameters()
    model.train()
    optim = torch.optim.SGD(model.parameters(), lr=.25, weight_decay=.0001)



    for epoch in range(args.num_epochs):
        train_dataset.seed(train_seeds[epoch])
        total_xent = 0
        total_tokens = 0
        for step, batch in enumerate(train_dataloader, 1):
            optim.zero_grad()
            model_state = model(batch)
            loss = loss_func(model_state, 
                             batch["target_output_features"]["tokens"],
                             batch["target_lengths"])
            loss.backward()
            #print(epoch, torch.exp(loss).item())
            num_tokens = batch["target_lengths"].sum().item()
            total_xent += loss.item() * num_tokens
            total_tokens += num_tokens
            optim.step()
            print("\rEpoch={} Step={}/{}  Avg. X-Entropy={:0.5f}".format(
                epoch, step, len(train_dataloader), total_xent / total_tokens),
                end="", flush=True)
            for p in model.parameters():
                assert torch.any(p.grad.ne(0))
        print()
        total_correct = 0
        total_tokens = 0
        for step, batch in enumerate(test_dataloader, 1):
            target = batch["target_output_features"]["tokens"]
            #print(target)
            model_state = model.greedy_decode(batch)
            prediction = model_state.get_result("output").t()
            #print(batch["target_input_features"]["tokens"])
            #print(prediction)
            steps = min(target.size(1), prediction.size(1))
            correct = target[:,:steps] == prediction[:,:steps]
            mask = target[:,:steps].eq(tgt_vocab.pad_index)
            correct.data.masked_fill_(mask, 0)
            total_tokens += (~mask).long().sum().item()
            total_correct += correct.long().sum().item()
            print("\rEpoch={} Step={}/{}  Acc.={:5.2f}%".format(
                epoch, step, len(test_dataloader), 
                100 * total_correct / total_tokens),
                end="", flush=True)

        print()




class CopyTaskWidget(object):
    def __init__(self):
        self._dataset_creator = DatasetCreatorWidget()
        self._model_creator = ModelCreatorWidget()
        self._model_trainer = ModelTrainerWidget(self._dataset_creator,
                                                 self._model_creator)

        self._main_widget = widgets.VBox([
            self._dataset_creator.display(),
            widgets.Label(" "),
            self._model_creator.display(),
            widgets.Label(" "),
            self._model_trainer.display(),
        ])

    def display(self):
        return self._main_widget


class ModelTrainerWidget(object):
    def __init__(self, dataset_creator, model_creator):
        self._dataset_creator = dataset_creator
        self._dataset_creator.set_model_trainer(self)

        self._model_creator = model_creator
        self._model_creator.set_model_trainer(self)
        self._widgets = {
            "models": widgets.Dropdown(options=[], value=None, 
                                       description="Select Model"),
            "datasets": widgets.Dropdown(options=[], value=None, 
                                         description="Select Dataset"),
            "batch_size": widgets.IntSlider(value=32, min=1, max=4096,
                                            step=1, 
                                            description='Batch Size:'),
            "num_epochs": widgets.IntSlider(value=5, min=1, max=2000,
                                            step=1, 
                                            description='Epochs:'),
            "lr": widgets.BoundedFloatText(value=0.1, min=0, max=100.0, 
                                           step=0.05,
                                           description='Learning Rate:'),
            "progress": widgets.IntProgress(value=10, min=0, max=10, step=1,
                                            description='Training:',
                                            bar_style=''),
            "progress_label": widgets.Label(),
 # 'success', 'info', 'warning', 'danger' or ''
            "results_plot": widgets.Output(),
        }

        button = widgets.Button(description="Train model!")
        button.on_click(self._action_hook)

        self._main_widget = widgets.VBox([
            widgets.Label("Train Model"),
            widgets.HBox([self._widgets["models"], self._widgets["datasets"]]),
            widgets.HBox([self._widgets["batch_size"], self._widgets["lr"]]),
            widgets.HBox([self._widgets["num_epochs"]]),
            button,           
            widgets.HBox([self._widgets["progress"], 
                          self._widgets["progress_label"]]),
            self._widgets["results_plot"],
        ])

    @property
    def model_params(self):
        return self._model_creator.get_model(self._widgets["models"].value)
        
    @property
    def datasets(self):                
        return self._dataset_creator.get_dataset(
            self._widgets["datasets"].value)

    @property
    def learning_rate(self):
        return self._widgets["lr"].value
        
    @property
    def batch_size(self):
        return self._widgets["batch_size"].value

    @property
    def num_epochs(self):
        return self._widgets["num_epochs"].value

    def update_models(self, new_models):
        self._widgets["models"].options = new_models.keys()
    
    def update_datasets(self, new_datasets):
        self._widgets["datasets"].options = new_datasets.keys()

    def display(self):
        return self._main_widget

    def _action_hook(self, b):
         
        model = self.create_model(self.model_params, self.datasets["train"])
        model.initialize_parameters()                                  
        
        train_data = Seq2SeqDataLoader(
            self.datasets["train"],
            model.encoder.embedding_context.named_vocabs,
            model.decoder.embedding_context.named_vocabs,
            batch_size=self.batch_size)
        test_data = Seq2SeqDataLoader(
            self.datasets["test"],
            model.encoder.embedding_context.named_vocabs,
            model.decoder.embedding_context.named_vocabs,
            batch_size=self.batch_size,
            include_original_data=True)
        loss_func = s2s.CrossEntropyLoss(
                pad_index=model.decoder.embedding_context.vocab.pad_index)
        optim = torch.optim.SGD(model.parameters(), lr=self.learning_rate, 
                                weight_decay=.0001)

        self.train_model(model, train_data, test_data, optim, loss_func)

    def train_model(self, model, train_data, val_data, optim, loss_func):
        
        progress = self._widgets["progress"]
        progress_label = self._widgets["progress_label"]
        progress_label.value = "Epoch 0/{}".format(self.num_epochs)
        progress.value = 0
        progress.max = self.num_epochs

        train_seeds = torch.LongTensor(self.num_epochs).random_(0,2**31)
        train_stats = {"x-entropy": []}

        #fig, axes = plt.subplots()
        for epoch in range(1, self.num_epochs + 1):
            train_data.dataset.seed(train_seeds[epoch - 1])
            train_xent = self.train_epoch(model, train_data, optim, loss_func)
            train_stats["x-entropy"].append(train_xent)
            progress.value += 1
            progress_label.value = "Epoch {}/{}  Avg. X-Entropy={:5.2f}"\
                .format(epoch, self.num_epochs, train_xent)
            with self._widgets["results_plot"]:

                plt.clf()
                plt.plot(train_stats["x-entropy"])
                plt.show()

    def train_epoch(self, model, data, optim, loss_func):        
        model.train()
        progress_label = self._widgets["progress_label"]
        epoch = self._widgets["progress"].value
        total_xent = 0
        total_tokens = 0
        for step, batch in enumerate(data, 1):
            optim.zero_grad()
            model_state = model(batch)
            loss = loss_func(model_state, 
                             batch["target_output_features"]["tokens"],
                             batch["target_lengths"])
            loss.backward()
            num_tokens = batch["target_lengths"].sum().item()
            total_xent += loss.item() * num_tokens
            total_tokens += num_tokens
            optim.step()
            progress_label.value = "Epoch {}/{}  Avg. X-Entropy={:5.2f}"\
                    .format(epoch, self.num_epochs,
                            total_xent / total_tokens) 
        return total_xent / total_tokens

    def create_model(self, model_params, train_dataset):
        word_list = train_dataset.word_list()
        src_vocab = ec.Vocab.from_word_list(word_list, pad="<PAD>", 
                                            unk="<UNK>", start="<START>") 
        tgt_vocab = ec.Vocab.from_word_list(word_list, pad="<PAD>",
                                            unk="<UNK>", start="<START>",
                                            stop="<STOP>")
        src_emb_ctx = ec.EmbeddingContext(
            src_vocab, embedding_size=model_params["embedding_dim"],
            name="tokens")
        tgt_emb_ctx = ec.EmbeddingContext(
            tgt_vocab, embedding_size=model_params["embedding_dim"],
            name="tokens")

        enc = s2s.RNNEncoder(src_emb_ctx, 
                             hidden_dim=model_params["hidden_dim"],
                             num_layers=model_params["rnn_layers"],
                             rnn_cell=model_params["rnn_cell"])

        dec = s2s.RNNDecoder(tgt_emb_ctx, 
                             hidden_dim=model_params["hidden_dim"],
                             num_layers=model_params["rnn_layers"],
                             rnn_cell=model_params["rnn_cell"],
                             attention=model_params["attention"])
        return s2s.EncoderDecoderBase(enc, dec)
                


class DatasetCreatorWidget(object):
    def __init__(self):
        self._datasets = OrderedDict()
        self._widgets = {
            "vocab_size": widgets.IntSlider(value=25, min=1, max=200000, 
                                            step=1, description='Vocab Size:'),
            "max_length": widgets.IntSlider(value=3, min=1, max=500, step=1,
                                            description='Max Length:'),
            "train_size": widgets.IntSlider(value=2048, min=1, max=20000,
                                            step=1, 
                                            description='Train Size:'),
            "test_size": widgets.IntSlider(value=100, min=1, max=20000, step=1,
                                           description='Test Size:'),
        }
            
        self._model_trainer = None
        button = widgets.Button(description="Create Dataset")
        button.on_click(self._action_hook)

        self._main_widget = widgets.VBox([
            widgets.Label("Create Dataset"), 
            widgets.HBox([self._widgets["vocab_size"], 
                          self._widgets["max_length"]]),
            widgets.HBox([self._widgets["train_size"], 
                          self._widgets["test_size"]]),
            button,
        ])

    def set_model_trainer(self, model_trainer):
        self._model_trainer = model_trainer

    def get_dataset(self, name):
        return self._datasets[name]

    @property
    def vocab_size(self):
        return self._widgets["vocab_size"].value

    @property
    def max_length(self):
        return self._widgets["max_length"].value

    @property
    def train_size(self):
        return self._widgets["train_size"].value

    @property
    def test_size(self):
        return self._widgets["test_size"].value

    def _action_hook(self, b):
        
        torch.manual_seed(7492023)
        test_seed = torch.LongTensor(1).random_(0,2**31).item()
        test_dataset = CopyDataset(self.vocab_size, self.max_length,
                                   self.test_size, random_seed=test_seed)
        train_dataset = CopyDataset(self.vocab_size, self.max_length,
                                    self.train_size)

        name = "dataset-{}".format(len(self._datasets) + 1)
        self._datasets[name] = {"train": train_dataset, "test": test_dataset}              
        if self._model_trainer:
            self._model_trainer.update_datasets(self._datasets) 

    def display(self):
        return self._main_widget
            
class ModelCreatorWidget(object):
    def __init__(self):

        self._models = {}
        self._widgets = {
                "datasets": widgets.Dropdown(options=[], value=None, 
                                             description="Select Dataset"),
                "emb_dim": widgets.IntSlider(value=100, min=1, max=1024,
                                             step=1, 
                                             description="Embedding Dim:"),
                "hidden_dim": widgets.IntSlider(value=100, min=1, max=1024,
                                                step=1, 
                                                description="Hidden Dim:"),
                "rnn_cell": widgets.Dropdown(options=["rnn", "gru", "lstm"],
                                             value="gru", 
                                             description="RNN Cell"),
                "rnn_layers": widgets.Dropdown(options=[1, 2, 3],
                                             value=1, 
                                             description="RNN Layers"),
                "attention": widgets.Dropdown(options=["none", "dot"],
                                              value="dot", 
                                              description="Attention"),

        }   
        
        self._model_trainer = None
        button = widgets.Button(description="Create Model")
        button.on_click(self._action_hook)
       
        self._main_widget = widgets.VBox([
            widgets.Label("Create Model"),
            widgets.HBox([self._widgets["emb_dim"],
                          self._widgets["hidden_dim"]]),
            widgets.HBox([self._widgets["rnn_cell"],
                          self._widgets["rnn_layers"],
                          self._widgets["attention"]]),
            button,
        ])

    def set_model_trainer(self, model_trainer):
        self._model_trainer = model_trainer

    def get_model(self, name):
        return self._models[name]

    def _action_hook(self, b):
        
        name = "model-{}".format(len(self._models) + 1)
        self._models[name] = {
            "embedding_dim": self._widgets["emb_dim"].value, 
            "hidden_dim": self._widgets["hidden_dim"].value,
            "rnn_cell": self._widgets["rnn_cell"].value,
            "rnn_layers": self._widgets["rnn_layers"].value,
            "attention": self._widgets["attention"].value
        }              

        if self._model_trainer:
                self._model_trainer.update_models(self._models)

    def display(self):
        return self._main_widget


if __name__ == "__main__":
    main()
