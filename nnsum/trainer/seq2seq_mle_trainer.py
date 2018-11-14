import torch
import torch.nn.functional as F

from ignite.engine import Engine, Events
from ignite._utils import _to_hours_mins_secs
from ignite.handlers import ModelCheckpoint

from nnsum.metrics import Loss
from nnsum.sequence_cross_entropy import sequence_cross_entropy

from colorama import Fore, Style
import ujson as json




def seq2seq_mle_trainer(model, optimizer, train_dataloader, max_epochs=10,
                        #validation_dataloader, max_epochs=10, pos_weight=None,
                        #summary_length=100, remove_stopwords=True,
                        grad_clip=5, gpu=-1, model_path=None):
                        #results_path=None, teacher_forcing=-1,
                        #create_trainer_fn=None):

    trainer = create_trainer(model, optimizer, grad_clip=grad_clip, gpu=gpu)

    xentropy = Loss(
        output_transform=lambda o: (o["total_xent"], o["total_tokens"]))
    xentropy.attach(trainer, "x-entropy")

    @trainer.on(Events.STARTED)
    def init_history(trainer):
        trainer.state.training_history = {"x-entropy": []}

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):

        xent = xentropy.compute()
        iterate = trainer.state.iteration
        msg = "Epoch[{}] Training {} / {}  X-Entropy: {:.3f}".format(
            trainer.state.epoch, iterate, len(train_dataloader),
            xent)
        if iterate < len(train_dataloader):
            print(msg, end="\r", flush=True)
        else:
            print(" " * len(msg), end="\r", flush=True)

    trainer.run(train_dataloader, max_epochs=max_epochs)

def create_trainer(model, optimizer, grad_clip=5, gpu=-1):

    pad_index = model.tgt_embedding_context.vocab.pad_index

    def _update(engine, batch):
        if gpu > -1: 
            _seq2seq2gpu(batch, gpu)
        model.train()
        optimizer.zero_grad()
        logits = model(batch)
        
        tgts = batch["target_output_features"]["tokens"].t()
        total_tokens = batch["target_lengths"].sum().item()

        tot_xent = sequence_cross_entropy(logits, tgts, pad_index=pad_index)
        avg_xent = tot_xent / total_tokens
        avg_xent.backward()
        
        for param in model.parameters():
            param.grad.data.clamp_(-grad_clip, grad_clip)
        optimizer.step()

        return {"total_xent": tot_xent, 
                "total_tokens": total_tokens}

    trainer = Engine(_update)
   
    return trainer



def _seq2seq2gpu(batch, gpu):
    sf = batch["source_features"]
    for feat in sf:
        sf[feat] = sf[feat].cuda(gpu)
    batch["source_lengths"] = batch["source_lengths"].cuda(gpu)
    batch["target_lengths"] = batch["target_lengths"].cuda(gpu)
    tif = batch["target_input_features"]
    for feat in tif:
        tif[feat] = tif[feat].cuda(gpu)
    tof = batch["target_output_features"]
    for feat in tof:
        tof[feat] = tof[feat].cuda(gpu)
