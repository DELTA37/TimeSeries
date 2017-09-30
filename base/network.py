import torch
import torch.nn as nn
from base.layers import *
from collections import OrderedDict


class BaseNet(nn.Module):
    def __init__(self, *args, trainable=True, restore=True, **kwargs):
        super(BaseNet, self).__init__(*args, **kwargs)
        self.trainable = trainable
        self.restore   = restore

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
    
    def get_criterion(self, params):
        pass

    def forward(self, inputs):
        pass

    def backward(self, grads):
        pass

    def get_inputs(self):
        return dict()
