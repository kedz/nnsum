import torch
import random

import nnsum
import logging
logging.getLogger().setLevel(logging.INFO)


def main():

    args = nnsum.fg_cli.training_argparser().parse_args()
    train_source = nnsum.data.RAMDataset(args.train_source)
    train_target = nnsum.data.RAMDataset(args.train_target)
    from nnsum import embedding_context
    x, y = embedding_context.create_vocab(
        train_source, features=["fields",],
        pad_token="<P>", unk_token="<U>", start_token="<S>") # "positions"])

    source_vocabs = [("tokens", x)] + [(k, v) for k, v in y.items()]
    source_embeddings = {}
    for feat, vocab in source_vocabs:
        ec = nnsum.module.EmbeddingContext(vocab, 100)
        source_embeddings[feat] = ec

    src_ec = nnsum.module.MultiEmbeddingContext(source_embeddings, merge="concat")

    tgt_vocab = embedding_context.create_vocab(
        train_target, pad_token="<P>", unk_token="<U>", 
        start_token="<S>", stop_token="</S>")
    tgt_ec = nnsum.module.EmbeddingContext(tgt_vocab, 200)

    from nnsum.model.rnn_seq2seq import RNNSeq2SeqModel
    target_vocabs = {"tokens": tgt_vocab}


    model = RNNSeq2SeqModel(src_ec, tgt_ec)
    model.cuda(0)


    source_vocabs = {k: v for k, v in source_vocabs}

    train_data = nnsum.data.AlignedDataset(train_source, train_target)
    train_loader = nnsum.data.Seq2SeqDataLoader(train_data, source_vocabs,
                                                target_vocabs,
                                                batch_size=16,
                                                pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    nnsum.trainer.seq2seq_mle_trainer(
        model, optimizer, train_loader, gpu=0)


    
    exit()

    

    random.seed(args["trainer"]["seed"])
    torch.manual_seed(args["trainer"]["seed"])
    torch.cuda.manual_seed_all(args["trainer"]["seed"])

    print(args["trainer"])
    print()
    print(args["emb"])
    print()
    print(args["enc"])
    print()
    print(args["ext"])

    print("Initializing vocabulary and embeddings.")
    embedding_context = nnsum.io.initialize_embedding_context(
        args["trainer"]["train_inputs"], **args["emb"])

    print("Loading training data.")
    train_data = nnsum.data.SummarizationDataset(
        embedding_context.vocab,
        args["trainer"]["train_inputs"],
        targets_dir=args["trainer"]["train_labels"],
        sentence_limit=args["trainer"]["sentence_limit"])
    train_loader = nnsum.data.SummarizationDataLoader(
        train_data, batch_size=args["trainer"]["batch_size"],
        num_workers=args["trainer"]["loader_workers"])

    print("Loading validation data.")
    val_data = nnsum.data.SummarizationDataset(
        embedding_context.vocab,
        args["trainer"]["valid_inputs"],
        targets_dir=args["trainer"]["valid_labels"],
        references_dir=args["trainer"]["valid_refs"],
        sentence_limit=args["trainer"]["sentence_limit"])
    val_loader = nnsum.data.SummarizationDataLoader(
        val_data, batch_size=args["trainer"]["batch_size"],
        num_workers=args["trainer"]["loader_workers"])

    if args["trainer"]["weighted"]:
        weight = nnsum.trainer.compute_class_weights(
            args["trainer"]["train_labels"],
            args["trainer"]["loader_workers"],
            sentence_limit=args["trainer"]["sentence_limit"])
    else:
        weight = None

    model = nnsum.cli.create_model_from_args(embedding_context, args)
    if args["trainer"]["gpu"] > -1:
        print("Placing model on device: {}".format(args["trainer"]["gpu"]))
        model.cuda(args["trainer"]["gpu"])
    model.initialize_parameters(logger=logging.getLogger())
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)

    nnsum.trainer.labels_mle_trainer(
        model, optimizer, train_loader, val_loader,
        pos_weight=weight, max_epochs=args["trainer"]["epochs"],
        summary_length=args["trainer"]["summary_length"],
        remove_stopwords=args["trainer"]["remove_stopwords"],
        gpu=args["trainer"]["gpu"],
        teacher_forcing=args["trainer"]["teacher_forcing"],
        model_path=args["trainer"]["model"],
        results_path=args["trainer"]["results"])

if __name__ == "__main__":
    main()
