import torch
import nnsum


def main():
    
    args = nnsum.fg_cli.training_parsers.seq2clf().parse_args()



    train_source = nnsum.data.RAMDataset(args.train_source)
    train_target = nnsum.data.RAMDataset(args.train_target)
    valid_source = nnsum.data.RAMDataset(args.valid_source)
    valid_target = nnsum.data.RAMDataset(args.valid_target)

    print(args)
    src_ec = nnsum.embedding_context.cli.from_args(
        args.MODS["emb"], train_source, pad_token="<P>", 
        unknown_token="<U>", start_token="<DS>", stop_token="</DS>",
        transpose=False)

    tgt_ec = nnsum.embedding_context.cli.label_context_from_args(
        args.MODS["lbl"], train_target)

    for class_, vocab in tgt_ec.named_vocabs.items():
        print(class_)
        for i, label in vocab.enumerate():
            print(i, label)
        print()

    encoders = []
    for label in args.MODS["lbl"].features: 
        encoder = nnsum.seq2clf.GatedCNNEncoder(
            src_ec.output_size,
            feature_maps=[200], window_sizes=[1])
        encoders.append(encoder)
    encoder = nnsum.seq2clf.ParallelEncoder(encoders)
    model = nnsum.seq2clf.SequenceClassifier(src_ec, encoder, tgt_ec)
    
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

    if True:
        label_counts = nnsum.trainer.get_label_counts(train_loader)
        label_weights = nnsum.trainer.get_balanced_weights(
            label_counts, gpu=args.gpu)

        print("Label Counts")
        print("------------")
        for cls, counts in label_counts.items():
            print(" {} :".format(cls))
            print("   {:30s}  {:8s} {:7s}".format("label", "count", 
                                                      "weight"))
            print("   {:30s}  {:8s} {:7s}".format(
                "=" * 30, "=" * 8, "=" * 7))
            for (label, count), w in zip(counts.items(), label_weights[cls]):
                print("   {:30s}  {:8.0f} {:5.4f}".format(label, count, w))
            print()
        print()

    if args.source_vocab is not None:
        args.source_vocab.parent.mkdir(exist_ok=True, parents=True)
        with args.source_vocab.open("wb") as fp:
            torch.save(model.source_embedding_context.named_vocabs, fp)

    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    nnsum.trainer.seq2clf_mle_trainer(
        model, optimizer, train_loader, valid_loader, gpu=args.gpu,
        model_path=args.model_path, max_epochs=50, 
        label_weights=label_weights,
        max_entropy_for_missing_data=args.max_entropy_for_missing_data,
        min_attention_entropy=args.min_attention_entropy,
        use_njsd_loss=args.use_njsd_loss)

if __name__ == "__main__":
    main()
