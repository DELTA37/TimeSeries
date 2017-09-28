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
            res = OrderedDict()
            for layer in dir(self):
                l = getattr(self, layer)
                if isinstance(l, Layer) or isinstance(l, BaseNet):
                    res.update(l.get_trainable())
            return res
        else:
            return OrderedDict()

    def get_restorable(self):
        if self.restore:
            res = OrderedDict()
            for layer in dir(self):
                l = getattr(self, layer)
                if isinstance(l, Layer) or isinstance(l, BaseNet):
                    res.update(l.get_restorable())
            return res
        else:
            return OrderedDict()


