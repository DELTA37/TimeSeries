import torch
import torch.nn as nn
from collections import OrderedDict

class Layer:
    def __init__(self, trainable, restore):
        self.trainable = trainable
        self.restore   = restore

    def get_trainable(self):
        if self.trainable:
            print(super(Layer, self))
            return nn.Module.state_dict(self)
        else:
            return OrderedDict()

    def get_restorable(self):
        if self.restore:
            return nn.Module.state_dict(self)
        else:
            return OrderedDict()

class Conv1dLayer(nn.Conv1d, Layer):
    def __init__(self, *args, trainable=True, restore=True, **kwargs):
        nn.Conv1d.__init__(self, *args, **kwargs)
        Layer.__init__(self, trainable, restore)

class Conv2dLayer(nn.Conv2d, Layer):
    def __init__(self, *args, trainable=True, restore=True, **kwargs):
        nn.Conv1d.__init__(self, *args, **kwargs)
        Layer.__init__(self, trainable, restore)

'''
etc
'''
