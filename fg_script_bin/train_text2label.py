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

    encoder = nnsum.seq2clf.CNNEncoder(src_ec.output_size) 
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
        num_workers=args.workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    nnsum.trainer.seq2clf_mle_trainer(
        model, optimizer, train_loader, valid_loader, gpu=args.gpu,
        model_path=args.model_path, max_epochs=50)

if __name__ == "__main__":
    main()
