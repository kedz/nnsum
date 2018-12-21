import torch
import nnsum


def main():

    args = nnsum.fg_cli.training_parsers.seq2seq().parse_args()
    print(args)
   
    train_source = nnsum.data.RAMDataset(args.train_source)
    train_target = nnsum.data.RAMDataset(args.train_target)
    valid_source = nnsum.data.RAMDataset(args.valid_source)
    valid_target = nnsum.data.RAMDataset(args.valid_target)

    # Construct model source embeddings. 
    src_ec = nnsum.embedding_context.cli.from_args(
        args.MODS["src-emb"], train_source, pad_token="<P>", 
        unknown_token="<U>", start_token="<ES>")
    
    # Construct model target embeddings.
    tgt_ec = nnsum.embedding_context.cli.from_args(
        args.MODS["tgt-emb"], train_target, pad_token="<P>", 
        unknown_token="<U>", start_token="<DS>", stop_token="</DS>")

    # Construct the model encoder decoder network.
    encoder = nnsum.seq2seq.cli.rnn_encoder_from_args(args.MODS["enc"], src_ec)
    decoder = nnsum.seq2seq.cli.rnn_decoder_from_args(args.MODS["dec"], tgt_ec)
    model = nnsum.seq2seq.EncoderDecoderModel(encoder, decoder)    

    model.initialize_parameters()

    if args.gpu > -1:
        model.cuda(args.gpu)

    train_loader = nnsum.data.Seq2SeqDataLoader(
        nnsum.data.AlignedDataset(train_source, train_target),
        src_ec.named_vocabs,
        tgt_ec.named_vocabs,
        batch_size=args.batch_size,
        pin_memory=args.gpu > -1,
        num_workers=args.workers,
        has_copy_attention=model.decoder._copy_attention_mode != "none")

    valid_loader = nnsum.data.Seq2SeqDataLoader(
        nnsum.data.AlignedDataset(valid_source, valid_target),
        src_ec.named_vocabs,
        tgt_ec.named_vocabs,
        batch_size=args.batch_size,
        pin_memory=args.gpu > -1,
        num_workers=args.workers,
        include_original_data=True,
        has_copy_attention=model.decoder._copy_attention_mode != "none")

    optimizer = nnsum.optimizer.cli.new_optimizer_from_args(
        args.MODS["opt"], model.parameters())
    opt_sch = nnsum.optimizer.cli.new_lr_scheduler_from_args(
        args.MODS["sch"], optimizer)
    nnsum.trainer.seq2seq_mle_trainer(
        model, opt_sch, train_loader, valid_loader, gpu=args.gpu,
        model_path=args.model_path, max_epochs=args.epochs)

if __name__ == "__main__":
    main()
