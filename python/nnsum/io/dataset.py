from .data_layout import DataLayout

import torch
from torch.autograd import Variable

import random
from collections import namedtuple


class Dataset(object):

    @staticmethod
    def _expand(data, layout, batch_size, shuffle, lengths, length_sort):
        return Dataset(*data, layout=layout, batch_size=batch_size, 
                       shuffle=shuffle, lengths=lengths, 
                       length_sort=length_sort)

    def __reduce__(self):
        items = (
            [dln for dln in zip(self.data_, self.length_, self.name_)],
            self.layout.layout_meta,
            self.batch_size,
            self.shuffle,
            self.example_lengths_,
            self.length_sort,
        )
        return (Dataset._expand, items)

    def __init__(self, *tensors, layout=None, batch_size=1, 
                 shuffle=True, gpu=-1, lengths=None, length_sort=True):
        
        self.data_ = []
        self.length_ = []
        self.name_ = []

        self.example_lengths_ = lengths

        self.cpu_buffers_ = []
        self.gpu_buffers_ = []
        self.name2cpu_buffer_ = {}
        self.name2gpu_buffer_ = {}
        
        self.shuffle_ = shuffle
        self.batch_size_ = batch_size
        self.length_sort = length_sort

        for tensor_data in tensors:
            if len(tensor_data) == 2:
                if isinstance(tensor_data[0], (list, tuple)):
                    self.register_list_data(tensor_data[0], tensor_data[1])
                else:
                    self.register_data(tensor_data[0], None, tensor_data[1])
            elif len(tensor_data) == 3:
                self.register_data(*tensor_data)
            else:
                raise Exception(
                    "tensors must be iterable of data, size, name.")

        if layout == None:
            layout = [[name, name] for name in self.name_]

        name2data = {name: data for name, data in zip(self.name_, self.data_)}
       
        self.data_layout_ = DataLayout(layout, name2data, "Dataset")
        self.cpu_batch_ = DataLayout(layout, self.name2cpu_buffer_, "CPUBatch")

        self.gpu = gpu


    def __getitem__(self, indexer):
        if isinstance(indexer, slice):
            index = [i for i in range(self.size)][indexer]
        else:
            index = [[i for i in range(self.size)][indexer]]
        return self.index_select(index)

    @property
    def layout(self):
        return self.data_layout_
        
    def index_select(self, index):
        if isinstance(index, (list, tuple)):
            index = torch.LongTensor(index)
        layout = self.data_layout_.layout_meta
        data = []
                
        for tensor, length, name in zip(self.data_, self.length_, self.name_):
            if isinstance(tensor, (list, tuple)):
                indexed_data = []
                for i in index:
                    indexed_data.append(tensor[i])
                data.append((indexed_data, name))
            else:

                if isinstance(length, (list, tuple)):
                    lengths = length
                    new_lengths = []
                    for l in lengths:
                        if l is not None:
                            new_lengths.append(l.index_select(0, index))
                        else:
                            new_lengths.append(l)
                    data.append((
                        tensor.index_select(0, index),
                        new_lengths,
                        name))

                elif length is not None:
                    data.append((
                        tensor.index_select(0, index),
                        length.index_select(0, index), 
                        name)) 

                else:
                    data.append((tensor.index_select(0, index), None, name))
            
        gpu = self.gpu
        batch_size = self.batch_size
        shuffle = self.shuffle

        lengths = None
        if self.example_lengths_ is not None:
            lengths = self.example_lengths_.index_select(0, index)

        return Dataset(*data, layout=layout, batch_size=batch_size,
                       shuffle=shuffle, gpu=gpu, lengths=lengths)

    def register_data(self, tensor, length, name):
        self.data_.append(tensor)
        self.length_.append(length)
        self.name_.append(name) 
        self.cpu_buffers_.append(Variable(tensor.new()))
        self.name2cpu_buffer_[name] = self.cpu_buffers_[-1]


    def register_list_data(self, list_data, list_name):
        self.data_.append(list_data)
        self.length_.append(None)
        self.name_.append(list_name) 
        self.cpu_buffers_.append([])
        self.name2cpu_buffer_[list_name] = self.cpu_buffers_[-1]


    def initialize_indices_(self):

        indices = [i for i in range(self.size)]
        if self.shuffle:
            random.shuffle(indices)
        indices = torch.LongTensor(indices)

        if self.example_lengths_ is not None and self.length_sort:
            step_size = self.batch_size * 25
            for i in range(0, self.size, step_size):
                indices_chunk = indices[i:i + step_size]
                lengths_batch = self.example_lengths_.index_select(
                    0, indices_chunk)
                sorted, sort_indices = torch.sort(
                    lengths_batch, dim=0, descending=False)
                indices_sorted_ = indices_chunk.index_select(0, sort_indices)
                indices[i:i + step_size] = indices_sorted_

                real_chunk_size = min(i + step_size, self.size)
                for j in range(i, real_chunk_size, self.batch_size):
                    local_desc = torch.from_numpy(
                        indices[j:j + self.batch_size].numpy()[::-1].copy())
                    indices[j:j + self.batch_size].copy_(local_desc)

        return indices

    def iter_batch(self):
        indices = self.initialize_indices_()

        for p in range(0, self.size, self.batch_size):
            indices_batch = indices[p:p + self.batch_size]

            for j in range(len(self.data_)):
                if isinstance(self.data_[j], (list, tuple)):
                    ldata = self.data_[j]
                    if self.gpu_ > -1:
                        ldata_buffer = self.gpu_buffers_[j]
                    else:
                        ldata_buffer = self.cpu_buffers_[j]
                    del ldata_buffer[:]

                    for idx in indices_batch:
                        ldata_buffer.append(ldata[idx])

                else:
                    length = self.length_[j]
                    if isinstance(length, (tuple, list)):
                        lengths = length
                        pre_buffer = self.data_[j]
                        for ax, length in enumerate(lengths, 1):
                            if length is None:
                                continue
                            max_len = length.index_select(
                                0, indices_batch).max()
                            pre_buffer = pre_buffer.narrow(ax, 0, max_len)
                        buffer = pre_buffer.index_select(
                            0, indices_batch, out=self.cpu_buffers_[j].data)

                    elif length is not None:
                        max_len = length.index_select(
                            0, indices_batch).max()
                        buffer = self.data_[j][:,:max_len].index_select(
                            0, indices_batch, out=self.cpu_buffers_[j].data)
                    else:
                        buffer = self.data_[j].index_select(
                            0, indices_batch, out=self.cpu_buffers_[j].data)

                    if self.gpu_ > -1:
                        self.gpu_buffers_[j].data.resize_(buffer.size()).copy_(
                            buffer)
            if self.gpu_ > -1:
                yield self.gpu_batch_
            else:
                yield self.cpu_batch_
 
    @property
    def shuffle(self):
        return self.shuffle_ 

    @shuffle.setter
    def shuffle(self, val):
        self.shuffle_ = bool(val)

    @property
    def gpu(self):
        return self.gpu_

    @gpu.setter
    def gpu(self, device):
        if not isinstance(device, int) or device < -1:
            raise Exception("device must be int >= -1")
        if device == -1:
            self.gpu_ = -1
            self.gpu_buffers_ = []
            self.name2gpu_buffer_ = {}
        else:
            self.gpu_ = device
            for name, buffer in zip(self.name_, self.cpu_buffers_):
                if isinstance(buffer, (list, tuple)):
                    gpu_buffer = []
                    self.gpu_buffers_.append(gpu_buffer)
                    self.name2gpu_buffer_[name] = gpu_buffer
                elif isinstance(buffer, Variable):
                    gpu_buffer = Variable(buffer.data.new().cuda(device))
                    self.gpu_buffers_.append(gpu_buffer)
                    self.name2gpu_buffer_[name] = gpu_buffer
                else:
                    raise Exception(
                        "data {} has unsupported type {}".format(
                            name, type(buffer)))
                                    
            self.gpu_batch_ = DataLayout(
                self.data_layout_.layout_meta, 
                self.name2gpu_buffer_, 
                "GPUBatch")

    @property
    def batch_size(self):
        return self.batch_size_
    
    @batch_size.setter
    def batch_size(self, batch_size):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise Exception("batch_size must be a positive integer.")
        self.batch_size_ = batch_size
        
    @property
    def size(self):
        if isinstance(self.data_[0], (list, tuple)):
            return len(self.data_[0])
        else:
            return self.data_[0].size(0)
 
    def __getattr__(self, attr):
        return getattr(self.data_layout_, attr)
