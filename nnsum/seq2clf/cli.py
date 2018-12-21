import argparse
from .cnn_encoder import CNNEncoder

def new_cnn_encoder_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-maps", default=[300], nargs="+", type=int)
    parser.add_argument("--window-sizes", default=[3], nargs="+", type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--activation", default="relu", 
                        choices=["relu", "tanh", "sigmoid"])
    return parser

def cnn_encoder_from_args(args, input_size):
    fm = args.feature_maps
    ws = args.window_sizes

    if len(fm) == 1 and len(ws) > 1:
        fm = fm * len(ws)
    elif len(ws) == 1 and len(fm) > 1:
        ws = ws * len(fm)

    if len(fm) != len(ws):
        raise Exception(
            "--feature-maps and --window-sizes have different numbers" \
            " of arguments.") 
    print("Initializing CNN encoder.")
    msg = " Feature Maps/Window Sizes: [{}]  Dropout: {:5.3f}  Activation: {}"
    print(msg.format(
        ", ".join(["{}/{}".format(f, w) for f, w in zip(fm, ws)]),
        args.dropout,
        args.activation))
    return CNNEncoder(input_size, feature_maps=fm, window_sizes=ws,
                      dropout=args.dropout, activation=args.activation)
