import torch
#import random

import nnsum
#import logging
#logging.getLogger().setLevel(logging.INFO)


def main():

    args = nnsum.fg_cli.training_parsers.seq2seq().parse_args()
   
    train_source = nnsum.data.RAMDataset(args.train_source)
    train_target = nnsum.data.RAMDataset(args.train_target)
    
    print(args)
    src_ec = nnsum.embedding_context.cli.from_args(
        args.MODS["src-emb"], train_source, pad_token="<P>", 
        unknown_token="<U>", start_token="<ES>")

    tgt_ec = nnsum.embedding_context.cli.from_args(
        args.MODS["tgt-emb"], train_target, pad_token="<P>", 
        unknown_token="<U>", start_token="<DS>", stop_token="</DS>")

    encoder = nnsum.seq2seq.cli.rnn_encoder_from_args(args.MODS["enc"], src_ec)
    decoder = nnsum.seq2seq.cli.rnn_decoder_from_args(args.MODS["dec"], tgt_ec)
    model = nnsum.seq2seq.EncoderDecoderModel(encoder, decoder)    

    if args.gpu > -1:
        model.cuda(args.gpu)

    train_loader = nnsum.data.Seq2SeqDataLoader(
        nnsum.data.AlignedDataset(train_source, train_target),
        src_ec.named_vocabs,
        tgt_ec.named_vocabs,
        batch_size=args.batch_size,
        pin_memory=args.gpu > -1,
        num_workers=args.workers)

    valid_source = nnsum.data.RAMDataset(args.valid_source)
    valid_target = nnsum.data.RAMDataset(args.valid_target)

    valid_loader = nnsum.data.Seq2SeqDataLoader(
        nnsum.data.AlignedDataset(valid_source, valid_target),
        src_ec.named_vocabs,
        tgt_ec.named_vocabs,
        batch_size=args.batch_size,
        pin_memory=args.gpu > -1,
        num_workers=args.workers)



    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    nnsum.trainer.seq2seq_mle_trainer(
        model, optimizer, train_loader, valid_loader, gpu=args.gpu,
        model_path=args.model_path, max_epochs=50)


    
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
