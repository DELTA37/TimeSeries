import torch
import torch.nn as nn
from collections import OrderedDict

class Layer:
    def __init__(self, trainable, restore):
        '''
        @param trainable : specify will we train this layer 
        @type  trainable : bool
        
        @param restore   : specify will we save and restore this layer
        @type  restore   : bool

        @return          : constructor
        @rtype           : None
        '''
        self.trainable = trainable
        self.restore   = restore

    def get_trainable(self):
        '''
        @return     : list of parameters which specified to train in this layer
        @rtype      : list(torch.nn.Parameter)
        '''
        if self.trainable:
            print(super(Layer, self))
            return nn.Module.named_parameters(self)
        else:
            return list()

    def get_restorable(self):
        '''
        @return     : list of parameters which specified to save and restore in this layer
        @rtype      : list(torch.nn.Parameter)
        '''
        if self.restore:
            return nn.Module.named_parameters(self)
        else:
            return list()


class Conv1dLayer(nn.Conv1d, Layer):
    def __init__(self, *args, trainable=True, restore=True, **kwargs):
        nn.Conv1d.__init__(self, *args, **kwargs)
        Layer.__init__(self, trainable, restore)

class Conv2dLayer(nn.Conv2d, Layer):
    def __init__(self, *args, trainable=True, restore=True, **kwargs):
        nn.Conv1d.__init__(self, *args, **kwargs)
        Layer.__init__(self, trainable, restore)

class LinearLayer(nn.Linear, Layer):
    def __init__(self, *args, trainable=True, restore=True, **kwargs):
        nn.Linear.__init__(self, *args, **kwargs)
        Layer.__init__(self, trainable, restore)

class SigmoidLayer(nn.Sigmoid, Layer):
    def __init__(self, *args, trainable=True, restore=True, **kwargs):
        nn.Sigmoid.__init__(self, *args, **kwargs)
        Layer.__init__(self, trainable, restore)

class MaxPool2dLayer(nn.MaxPool2d, Layer):
    def __init__(self, *args, trainable=True, restore=True, **kwargs):
        nn.MaxPool2d.__init__(self, *args, **kwargs)
        Layer.__init__(self, trainable, restore)

class ReLULayer(nn.ReLU, Layer):
    def __init__(self, *args, trainable=True, restore=True, **kwargs):
        nn.ReLU.__init__(self, *args, **kwargs)
        Layer.__init__(self, trainable, restore)


class Conv2d_BN_ReLULayer(nn.Module, Layer):
    def __init__(self, in_channels, out_channels, kernel_size, *args, 
            trainable=True, restore=True, 
            stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        nn.Module.__init__(self, *args, **kwargs)
        Layer.__init__(self, trainable, restore)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

Layer.AccessableMethods = dict({cl.__name__ : cl for cl in Layer.__subclasses__()})
