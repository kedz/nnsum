import sys
import argparse
from collections import OrderedDict, defaultdict


class ModuleCLIParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super(ModuleCLIParser, self).__init__(*args, **kwargs)
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
            main_args = super(ModuleCLIParser, self).parse_args(args=args)
        else:
            main_args = super(ModuleCLIParser, self).parse_args(
                args=args[:subsections[0]])

        main_args.MODS = {}        
        mod2args = defaultdict(list)
        for i, start in enumerate(subsections):
            mod = args[start]
            if i + 1 < len(subsections):
                mod_args = args[start + 1:subsections[i+1]]
            else:
                mod_args = args[start + 1:]
            mod2args[mod] = mod_args
        for mod, parser in self._module_clis.items():
            main_args.MODS[mod[2:]] = parser.parse_args(args=mod2args[mod])

        return main_args
