import torch
from nnsum.datasets import CopyDataset
import nnsum.embedding_context as ec
import nnsum.seq2seq as s2s
from nnsum.data import Seq2SeqDataLoader


def make_dataset(dataset_params):
    train_data = CopyDataset(dataset_params["vocab_size"], 
                             dataset_params["min_length"],
                             dataset_params["max_length"],
                             dataset_params["train_size"],
                             random_seed=dataset_params["train_seed"])
    test_data = CopyDataset(dataset_params["vocab_size"], 
                            dataset_params["min_length"],
                            dataset_params["max_length"],
                            dataset_params["test_size"],
                            random_seed=dataset_params["test_seed"])

    return train_data, test_data
    
def make_model(model_params, train_dataset):
        word_list = train_dataset.word_list()
        
        src_n_unk = int(len(word_list) * model_params["source_unknown_rate"])
        tgt_n_unk = int(len(word_list) * model_params["target_unknown_rate"])

        src_word_list = word_list[:-src_n_unk] if src_n_unk > 0 else word_list
        tgt_word_list = word_list[:-tgt_n_unk] if tgt_n_unk > 0 else word_list

        src_vocab = ec.Vocab.from_word_list(src_word_list, pad="<PAD>", 
                                            unk="<UNK>", start="<START>") 
        tgt_vocab = ec.Vocab.from_word_list(tgt_word_list, pad="<PAD>",
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

        if model_params["architecture"] == "Seq2Seq":
            dec = s2s.RNNDecoder(tgt_emb_ctx, 
                                 hidden_dim=model_params["hidden_dim"],
                                 num_layers=model_params["rnn_layers"],
                                 rnn_cell=model_params["rnn_cell"],
                                 attention=model_params["attention"])
        else:
            print("Radical!")
            dec = s2s.PointerGeneratorDecoder(
                tgt_emb_ctx, 
                hidden_dim=model_params["hidden_dim"],
                num_layers=model_params["rnn_layers"],
                rnn_cell=model_params["rnn_cell"],
                attention=model_params["attention"])

        model = s2s.EncoderDecoderBase(enc, dec)
        model.initialize_parameters()
        return model
 
def make_optimizer(optimizer_params, model):
    params = {key: value for key, value in optimizer_params.items()
              if key not in ["name", "type"]}
    constr = getattr(torch.optim, optimizer_params["type"])
    return constr(model.parameters(), **params)

def start_train_run(dataset_params, model_params, optimizer_params,
                    trainer_params, train_complete_callback=None,
                    valid_complete_callback=None):
   
    train_data, test_data = make_dataset(dataset_params)
    model = make_model(model_params, train_data) 
    optimizer = make_optimizer(optimizer_params, model)

    train_batches = Seq2SeqDataLoader(
        train_data,
        model.encoder.embedding_context.named_vocabs,
        model.decoder.embedding_context.named_vocabs,
        batch_size=trainer_params["batch_size"])
    test_batches = Seq2SeqDataLoader(
        test_data,
        model.encoder.embedding_context.named_vocabs,
        model.decoder.embedding_context.named_vocabs,
        batch_size=trainer_params["batch_size"])

    pad_index = model.decoder.embedding_context.vocab.pad_index
    arch = model_params["architecture"]
    if model_params["architecture"] == "Seq2Seq":
        loss_func = s2s.CrossEntropyLoss(pad_index=pad_index)
    else:
        loss_func = s2s.PointerGeneratorCrossEntropyLoss(pad_index=pad_index)

    for epoch in range(1, trainer_params["epochs"] + 1):

        train_stats = train_epoch(model, train_batches, optimizer, loss_func)
        if train_complete_callback:
            train_complete_callback(train_stats)

        valid_stats = valid_epoch(model, test_batches, loss_func, arch)
        if valid_complete_callback:
            valid_complete_callback(valid_stats)

    return {"model": model, "model_params": model_params, 
            "dataset_params": dataset_params, 
            "optimizer_params": optimizer_params,
            "trainer_params": trainer_params}

def train_epoch(model, batches, optim, loss_func):

    model.train()
    total_xent = 0
    total_tokens = 0
    for step, batch in enumerate(batches, 1):
        optim.zero_grad()
        model_state = model(batch)
        loss = loss_func(model_state, batch)
        loss.backward()
        num_tokens = batch["target_lengths"].sum().item()
        total_xent += loss.item() * num_tokens
        total_tokens += num_tokens
        optim.step()
        print("Steps {}/{}  Avg. X-Entropy={:5.2f}".format(
            step, len(batches), total_xent / total_tokens),
            end="\r", flush=True)

    return {"x-entropy": total_xent / total_tokens}

def valid_epoch(model, batches, loss_func, arch):
    model.eval()
    total_xent = 0
    total_tokens = 0
    total_correct = 0
    for step, batch in enumerate(batches, 1):
        model_state = model(batch)
        loss = loss_func(model_state, batch)

        pred_output = model.greedy_decode(batch).get_result("output").t()
        if arch == "Pointer-Generator":
            total_correct += get_correct(
                pred_output, batch["copy_targets"],
                model.decoder.embedding_context.vocab.pad_index)
        else:
            total_correct += get_correct(
                pred_output, batch["target_output_features"]["tokens"],
                model.decoder.embedding_context.vocab.pad_index)

        num_tokens = batch["target_lengths"].sum().item()
        total_xent += loss.item() * num_tokens
        total_tokens += num_tokens

        print("Steps {}/{}  Avg. X-Entropy={:5.2f}  Acc.={:5.2f}%".format(
            step, len(batches), total_xent / total_tokens, 
            total_correct / total_tokens * 100),
            end="\r", flush=True)
    return {"x-entropy": total_xent / total_tokens,
            "accuracy": total_correct / total_tokens}

def get_correct(pred_output, gold_output, pad_index):
        steps = min(pred_output.size(1), gold_output.size(1))
        mask = gold_output[:,:steps].eq(pad_index)
        correct = pred_output[:,:steps] == gold_output[:,:steps]
        return correct.masked_fill(mask,0).long().sum().item()


