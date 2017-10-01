import torch
import torch.nn as nn
from base.layers import *
from collections import OrderedDict
from abc import *

class BaseNet(nn.Module):
    def __init__(self, params, *args, trainable=True, restore=True, **kwargs):
        super(BaseNet, self).__init__(*args, **kwargs)
        self.trainable  = trainable
        self.restore    = restore
        self.batch_size = [params['batch_size'],]

    def get_trainable(self):
        if self.trainable:
            res = list()
            for layer in dir(self):
                l = getattr(self, layer)
                if isinstance(l, Layer) or isinstance(l, BaseNet):
                    tr = l.get_trainable()
                    newtr = []
                    for key, val in tr:
                        newtr.append((layer + '.' + key, val))
                    res += newtr
            return res
        else:
            return list()

    def get_restorable(self):
        if self.restore:
            res = list()
            for layer in dir(self):
                l = getattr(self, layer)
                if isinstance(l, Layer) or isinstance(l, BaseNet):
                    rs = l.get_restorable()
                    newrs = []
                    for key, val in rs:
                        newrs.append((layer + '.' + key, val))
                    res += newrs
            return res
        else:
            return list()
   
    @abstractmethod
    def get_criterion(self, params):
        pass 

    def __call__(self, x):
        self.once = 0
        for i in range(len(self.get_outputs().values())): 
            nn.Module.__call__(self, x)
        return self.stored

    def forward(self, inputs):
        self.once += 1 
        if self.once == 1:
            self.stored = self.dict_forward(inputs)
        return list(self.stored.values())[self.once - 1]
    
    @abstractmethod
    def dict_forward(self, inputs):
        pass

    @abstractmethod
    def get_inputs(self):
        pass

    @abstractmethod
    def get_outputs(self):
        pass
