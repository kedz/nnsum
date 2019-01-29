import argparse

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

if __name__ == "__main__":
    main()
