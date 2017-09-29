import torch
import torch.nn as nn
from base.layers import *
from collections import OrderedDict

def rename(self,key,new_key): # popular method for OrderedDict
    self[new_key] = self[key]
    del self[key]

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
                    tr = l.get_trainable()
                    lst = list(tr.keys())
                    for key in lst: 
                        rename(tr, key, layer + '.' + key)
                    res.update(tr)
            return res
        else:
            return OrderedDict()

    def get_restorable(self):
        if self.restore:
            res = OrderedDict()
            for layer in dir(self):
                l = getattr(self, layer)
                if isinstance(l, Layer) or isinstance(l, BaseNet):
                    rs = l.get_restorable()
                    lst = list(rs.keys())
                    for key in lst: 
                        rename(rs, key, layer + '.' + key)
                    res.update(rs)

            return res
        else:
            return OrderedDict()
    
    def get_loss(self, params):
        pass
