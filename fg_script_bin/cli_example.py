import nnsum
import argparse
from collections import OrderedDict

import sys

class ModuleArgumentParser(argparse.ArgumentParser):

    def __init__(self):
        super(ModuleArgumentParser, self).__init__()
        self._module_clis = OrderedDict()

    def add_module_cli(self, name, subparser):
        if not name.startswith("--"):
            name = "--" + name
        self._module_clis[name] = subparser

    def parse_args(self, args=None):
        if args is None:
            args = sys.argv[1:]
        subsections = [i for i, arg in enumerate(args) 
                       if arg in self._module_clis]
        if len(subsections) == 0:
            return super(ModuleArgumentParser, self).parse_args(args=args[1:])
        
        main_args = super(ModuleArgumentParser, self).parse_args(
            args=args[:subsections[0]])
        main_args.MODS = {}
        for i, start in enumerate(subsections):
            mod = args[start]
            if i + 1 < len(subsections):
                mod_args = args[start + 1:subsections[i+1]]
            else:
                mod_args = args[start + 1:]
            mod_args = self._module_clis[mod].parse_args(args=mod_args)
            main_args.MODS[mod[2:]] = mod_args
        return main_args
#        args = super(ModuleArgumentParser, self).parse_known_args(args=args)
#        print(args)

main_parser = ModuleArgumentParser()
emb_parser = argparse.ArgumentParser()
emb_parser.add_argument("--dims", type=int, required=True)

mlp_parser = argparse.ArgumentParser()
mlp_parser.add_argument("--dims", type=int, nargs="+", required=True)
main_parser.add_argument("--opt1", choices=["a", "b", "c"], required=True)
main_parser.add_module_cli("emb", emb_parser)
main_parser.add_module_cli("mlp", mlp_parser)

main_parser.parse_args()




exit()
parser = argparse.ArgumentParser()
subparsers1 = parser.add_subparsers(help='types of A')
B1 = subparsers1.add_parser("B")
B1.add_argument("--b", nargs=1, required=True)

A1 = subparsers1.add_parser("A")
A1.add_argument("--a", nargs=1, required=True)

subparsers2 = parser.add_subparsers(help='types of Z')
C1 = subparsers1.add_parser("C")
C1.add_argument("--c", nargs=1, required=True)
D1 = subparsers1.add_parser("D")
D1.add_argument("--d", nargs=1, required=True)
args = parser.parse_args()

print(args)
