import torch
import torch.nn as nn
from collections import OrderedDict
from functools import reduce


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


class LayerList(Layer):
    def __init__(self, trainable, restore, *args):
        '''
        @param trainable : specify will we train this layer 
        @type  trainable : bool
        
        @param restore   : specify will we save and restore this layer
        @type  restore   : bool

        @return          : constructor
        @rtype           : None
        '''
        super(LayerList, self).__init__(trainable, restore)
        self.modules   = args

    def get_trainable(self):
        '''
        @return     : list of parameters which specified to train in this layer
        @rtype      : list(torch.nn.Parameter)
        '''
        if self.trainable:
            lst = []
            for module in self.modules:
                lst.append(module.get_trainable())
            return lst
        else:
            return list()

    def get_restorable(self):
        '''
        @return     : list of parameters which specified to save and restore in this layer
        @rtype      : list(torch.nn.Parameter)
        '''
        if self.restore:
            lst = []
            for module in self.modules:
                lst.append(module.get_restorable())
            return lst
        else:
            return list()

class Conv1dLayer(nn.Conv1d, Layer):
    def __init__(self, *args, trainable=True, restore=True, **kwargs):
        nn.Conv1d.__init__(self, *args, **kwargs)
        Layer.__init__(self, trainable, restore)

class Conv2dLayer(nn.Conv2d, Layer):
    def __init__(self, *args, trainable=True, restore=True, **kwargs):
        nn.Conv2d.__init__(self, *args, **kwargs)
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

class RNNLayer(nn.RNN, Layer):
    def __init__(self, *args, trainable=True, restore=True, **kwargs):
        nn.RNN.__init__(self, *args, **kwargs)
        Layer.__init__(self, trainable, restore)

class LSTMLayer(nn.LSTM, Layer):
    def __init__(self, *args, trainable=True, restore=True, **kwargs):
        nn.LSTM.__init__(self, *args, **kwargs)
        Layer.__init__(self, trainable, restore)


class BN1dLayer(nn.BatchNorm1d, Layer):
     def __init__(self, *args, trainable=True, restore=True, **kwargs):
        nn.BatchNorm1d.__init__(self, *args, **kwargs)
        Layer.__init__(self, trainable, restore)
   
class LeakyReLULayer(nn.LeakyReLU, Layer):
     def __init__(self, *args, trainable=True, restore=True, **kwargs):
        nn.LeakyReLU.__init__(self, *args, **kwargs)
        Layer.__init__(self, trainable, restore)

class SoftMaxLayer(nn.Softmax, Layer):
     def __init__(self, *args, trainable=True, restore=True, **kwargs):
        nn.Softmax.__init__(self, *args, **kwargs)
        Layer.__init__(self, trainable, restore)


class Conv2d_BN_ReLULayer(nn.Module, Layer):
    def __init__(self, in_channels, out_channels, kernel_size, trainable=True, restore=True, 
            stride=1, padding=0, dilation=1, groups=1, bias=True):
        nn.Module.__init__(self)
        Layer.__init__(self, trainable, restore)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module, LayerList):
    def __init__(self, trainable, restore, *args):
        nn.Module.__init__(self)
        LayerList.__init__(self, trainable, restore, args)
    
    def forward(self, x):
        lst = [module(x) for module in self.modules]
        return reduce(lambda x, y: x + y, lst)


class DenseBlock(nn.Module, LayerList):
    def __init__(self, trainable, restore, *args):
        nn.Module.__init__(self)
        LayerList.__init__(self, trainable, restore, args)
    
    def forward(self, x):
        lst = [module(x) for module in self.modules]
        return torch.cat(lst, dim=-1)

Layer.AccessableMethods = dict({cl.__name__ : cl for cl in Layer.__subclasses__()})
