import torch
from collections import defaultdict
from .parameterized import build_object


class HParamParser(object):
    def __init__(self):
        self._uri_object_cache = {}
        self._uri_getter_cache = defaultdict(list)

    def parse(self, hparams):
        parsed_hparams = self._parser_helper(hparams)    

        for uri, obj in self._uri_object_cache.items():
            getters = self._uri_getter_cache[uri]
            self._set_resources(obj, getters, parsed_hparams)

        return self._build_objects(parsed_hparams)

    def _build_objects(self, hparams):
        if isinstance(hparams, dict):
                
            new_dict = {}
            for key, value in hparams.items():
                new_dict[key] = self._build_objects(value)
            
            if "__modulename__" in new_dict:
                return build_object(new_dict)
            else:
                return new_dict

        elif isinstance(hparams, (list, tuple)):
            return [self._build_objects(hparam) for hparam in hparams]
        else:
            return hparams       

    def _set_resources(self, obj, getters, parsed_hparams):
        for getter in getters:
            item = parsed_hparams
            while len(getter) > 1:
                item = item[getter.pop(0)]
            item[getter[0]] = obj

    def _resolve_resource(self, args, getter):
        uri = args.get("uri", None)
        if uri is None:
            raise Exception(
                "uri is not set for resource at: {}".format(str(getter)))
        
        if not uri.startswith("file://"):
            raise Exception("Non-file uris not implemented: {}".format(uri))

        if not uri.endswith(".pth"):
            raise Exception("Non .pth files not supported: {}".format(uri))

        if uri not in self._uri_object_cache:
            obj = torch.load(uri[7:])
            self._uri_object_cache[uri] = obj
            
        self._uri_getter_cache[uri].append(getter)

    def _parser_helper(self, hparams, getter=[]):    
        if isinstance(hparams, dict):

            if hparams.get("__modulename__", None) == "resource":
                self._resolve_resource(hparams, getter)
                return hparams
                
            new_dict = {}
            for key, value in hparams.items():
                new_dict[key] = self._parser_helper(value, getter + [key])
            
            return new_dict

        elif isinstance(hparams, (list, tuple)):
            return [self._parser_helper(hparam, getter + [i]) 
                    for i, hparam in enumerate(hparams)]
        else:
            return hparams
