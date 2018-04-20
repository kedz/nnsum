from collections import namedtuple

# TODO make work with dict layout and make todict use ordered dict

class DataLayout(object):

    @staticmethod
    def _expand(layout_meta, label2data, root_name):
        return DataLayout(layout_meta, label2data, root_name=root_name)

    def __reduce__(self):
        return (DataLayout._expand, 
                (self.layout_meta, self.label2data, self.root_name))

    def __init__(self, layout_meta, label2data, root_name="dataset"):
        
        self.layout_meta_ = layout_meta
        self.label2data_ = label2data
        self.root_name_ = root_name
        self.replacement_sites_ = {} # TODO can probably remove this
        self.layout_ = self.recursive_layout_init_(layout_meta, root_name)
        

    def __len__(self):
        for val in self.label2data_.values():
            if isinstance(val, (list, tuple)):
                return len(val)
            else:
                return val.size(0)

    def recursive_layout_init_(self, ld, name):
        attributes = []
        values = []
        for key, value in ld:
            if isinstance(value, (list,tuple)):
                attributes.append(key)
                ds = self.recursive_layout_init_(value, key)
                values.append(ds)
            else:
                attributes.append(key)
                values.append(self.label2data_[value])
        ntc = namedtuple(name, attributes)
        # make safe for pickling 
        globals()[ntc.__name__] = ntc
        return ntc(*values) 

    def __iter__(self):
        for item in self.layout_:
            yield item

    def __getitem__(self, item):
        return self.layout_[item]

    def index_select(self, index):
        idx_label2data = {}
        for label, data in self.label2data_.items():
            idx_label2data[label] = data.index_select(0, index)
        return DataLayout(
            self.layout_meta, idx_label2data, root_name=self.root_name_)

    def to_dict_helper_(self, t):
        
        if hasattr(t, "_fields"):
            result = {}
            for field in t._fields:
                result[field] = self.to_dict_helper_(getattr(t, field))
            return result
        else:
            return t

    def to_dict(self):
        d = {}
        for field in self.layout_._fields:
            d[field] = self.to_dict_helper_(getattr(self.layout_, field))
        return d

    @property
    def root_name(self):
        return self.root_name_

    @property
    def label2data(self):
        return self.label2data_

    @property
    def layout_meta(self):
        return self.layout_meta_

    def __getattr__(self, key):
        return getattr(self.layout_, key)
