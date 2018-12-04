import argparse
from .rnn_encoder import RNNEncoder
from .rnn_decoder import RNNDecoder

def new_rnn_encoder_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--rnn-cell", choices=["lstm", "gru", "rnn"], 
                        default="gru")
    return parser

def rnn_encoder_from_args(args, embedding_context):
    return RNNEncoder(embedding_context, hidden_dim=args.hidden_dim, 
                      num_layers=args.num_layers, rnn_cell=args.rnn_cell)

def new_rnn_decoder_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--rnn-cell", choices=["lstm", "gru", "rnn"], 
                        default="gru")
    parser.add_argument("--attention", choices=["dot", "none"],
                        default="none", type=str)
    return parser

def rnn_decoder_from_args(args, embedding_context):
    return RNNDecoder(embedding_context, hidden_dim=args.hidden_dim, 
                      num_layers=args.num_layers, rnn_cell=args.rnn_cell,
                      attention=args.attention)


