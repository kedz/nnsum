import argparse
import torch
from .scheduler import NoScheduler, DecreaseOnPlateauScheduler


def new_optimizer_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--lr", type=float, default=1e-4, required=False)
    parser.add_argument("--l2", type=float, default=1e-4, required=False)
    parser.add_argument("--momentum", type=float, default=0., required=False)
    return parser

def new_optimizer_from_args(args, parameters):
    if args.alg == "sgd":
        optimizer = torch.optim.SGD(
            parameters, lr=args.lr, momentum=args.momentum, 
            weight_decay=args.l2)

    else:
        optimizer = torch.optim.Adam(
            parameters, lr=args.lr, 
            weight_decay=args.l2)

    return optimizer    

def new_lr_scheduler_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", choices=["decrease-on-plateau", "none"],
                        default="none", type=str)
    parser.add_argument("--metric", choices=["x-entropy", "accuracy", "f1",
                                             "bleu", "nist"],
                        default="f1", type=str)
    parser.add_argument(
        "--decay-factor", default=1e-1, type=float, required=False)
    parser.add_argument(
        "--patience", default=10, type=int, required=False)
    return parser

def new_lr_scheduler_from_args(args, optimizer):
    if args.scheduler == "none":
        return NoScheduler(optimizer, args.metric)
    else:
        return DecreaseOnPlateauScheduler(
            optimizer, args.metric, decay_factor=args.decay_factor,
            patience=args.patience)
