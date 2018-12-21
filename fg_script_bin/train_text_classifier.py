import torch
import nnsum


def main():
    
    args = nnsum.fg_cli.training_parsers.cnn_seq2clf().parse_args()
    print(args)

    train_source = nnsum.data.RAMDataset(args.train_source)
    train_target = nnsum.data.RAMDataset(args.train_target)
    valid_source = nnsum.data.RAMDataset(args.valid_source)
    valid_target = nnsum.data.RAMDataset(args.valid_target)

    # Construct model input embedding contexts.
    src_ec = nnsum.embedding_context.cli.from_args(
        args.MODS["emb"], train_source, pad_token="<P>", 
        unknown_token="<U>", start_token="<DS>", stop_token="</DS>",
        transpose=False)

    # Construct model CNN encoder.
    encoder = nnsum.seq2clf.cli.cnn_encoder_from_args(
        args.MODS["enc"], src_ec.output_size)
    
    # Construct model target embedding contexts.
    tgt_ec = nnsum.embedding_context.cli.label_context_from_args(
        args.MODS["lbl"], train_target, encoder.output_size)
    
    model = nnsum.seq2clf.SequenceClassifier(src_ec, encoder, tgt_ec)

    if args.balance_weights:
        class_weights = nnsum.trainer.get_balanced_weights(
            tgt_ec.label_frequencies(), gpu=args.gpu)

        for class_, vocab in tgt_ec.named_vocabs.items():
            print(class_)
            for i, label in vocab.enumerate():
                print("{:5d} {:7d} {:5.3f} {}".format(
                    i, vocab.count(label), class_weights[class_][i].item(), 
                    label))
            print()
    else:
        class_weights = None
        for class_, vocab in tgt_ec.named_vocabs.items():
            print(class_)
            for i, label in vocab.enumerate():
                print("{:5d} {:7d} {:5.3f} {}".format(
                    i, vocab.count(label), 1.0, 
                    label))
            print()
   
    model.initialize_parameters()
    
    if args.gpu > -1:
        model.cuda(args.gpu)

    train_loader = nnsum.data.Seq2ClfDataLoader(
        nnsum.data.AlignedDataset(train_source, train_target),
        src_ec.named_vocabs,
        tgt_ec.named_vocabs,
        batch_size=args.batch_size,
        pin_memory=args.gpu > -1,
        num_workers=args.workers)

    valid_loader = nnsum.data.Seq2ClfDataLoader(
        nnsum.data.AlignedDataset(valid_source, valid_target),
        src_ec.named_vocabs,
        tgt_ec.named_vocabs,
        batch_size=args.batch_size,
        pin_memory=args.gpu > -1,
        shuffle=False,
        num_workers=args.workers)

    if args.source_vocab is not None:
        args.source_vocab.parent.mkdir(exist_ok=True, parents=True)
        with args.source_vocab.open("wb") as fp:
            torch.save(model.source_embedding_context.named_vocabs, fp)

    optimizer = nnsum.optimizer.cli.new_optimizer_from_args(
        args.MODS["opt"], model.parameters())
    opt_sch = nnsum.optimizer.cli.new_lr_scheduler_from_args(
        args.MODS["sch"], optimizer)

    nnsum.trainer.seq2clf_mle_trainer(
        model, opt_sch, train_loader, valid_loader, gpu=args.gpu,
        model_path=args.model_path, max_epochs=args.epochs,
        label_weights=class_weights)

if __name__ == "__main__":
    main()
