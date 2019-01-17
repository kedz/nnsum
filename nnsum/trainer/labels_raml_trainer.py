from .labels_mle_trainer import labels_mle_trainer

import torch
import torch.nn.functional as F

from ignite.engine import Engine, Events

def labels_raml_trainer(model, optimizer, train_loader,
                       validation_loader, max_epochs=10, pos_weight=None,
                       summary_length=100, remove_stopwords=True,
                       grad_clip=5, gpu=-1, model_path=None, 
                       results_path=None, teacher_forcing=-1,
                       create_trainer_fn=None):

    labels_mle_trainer(model, optimizer, train_loader, validation_loader,
                       max_epochs=max_epochs, pos_weight=pos_weight,
                       summary_length=summary_length,
                       remove_stopwords=remove_stopwords, grad_clip=grad_clip,
                       gpu=gpu, model_path=model_path,
                       results_path=results_path,
                       teacher_forcing=teacher_forcing,
                       create_trainer_fn=create_trainer)

def create_trainer(model, optimizer, pos_weight=None, grad_clip=5, gpu=-1):

    def _update(engine, batch):
        model.train()
        batch = batch.to(gpu)
        optimizer.zero_grad()
        logits = model(
            batch, decoder_supervision=batch.targets.float())
        mask = batch.targets.gt(-1).float()
        total_sentences_batch = int(batch.num_sentences.data.sum())
        
        if pos_weight is not None:
            mask.data.masked_fill_(batch.targets.data.eq(1), pos_weight)

        bce = F.binary_cross_entropy_with_logits(
            logits.unsqueeze(1).repeat(1,batch.targets.size(1),1), batch.targets.float(),
            weight=mask, 
            reduction='none')
        bce = (bce.sum(2) * batch.scores).sum(1).sum(0)

        avg_bce = bce / float(total_sentences_batch)
        avg_bce.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-grad_clip, grad_clip)
        optimizer.step()

        return {"total_xent": bce, 
                "total_examples": total_sentences_batch}

    trainer = Engine(_update)
   
    return trainer


