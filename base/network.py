import torch
import torch.nn as nn
from base.layers import *
from collections import OrderedDict
from abc import *
import base.deprecater as depr

class BaseNet(nn.Module):
    def __init__(self, params, *args, trainable=True, restore=True, **kwargs):
        '''
        @param     trainable            : if trainable = False, then it's learning rate = 0
        @type      trainable            : bool

        @param     restore              : if restore = True, then this net will record into file and can be read from file
        @type      restore              : bool

        @param     params               : configuration of our network
        @type      params               : dict

        @param     params['batch_size'] : batch size 
        @type      params['batch_size'] : int

        @return                         : constructor
        @rtype                          : None 
        '''
        super(BaseNet, self).__init__(*args, **kwargs)
        self.trainable  = trainable
        self.restore    = restore
        self.params     = depr.DeprecateWrapper(params, params['deprecate'])

    def get_trainable(self):
        '''
        @return                         : list of all variables which we want to train
        @rtype                          : list
        '''
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
        '''
        @return                         : list of all variables which we want to save and restore
        @rtype                          : list
        '''
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
        '''
        @param      params : specify loss parameters
        @type       params : dict()

        @return            : loss function for model
        @rtype             : callable
        '''
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

    def get_outputs(self):
        '''
        @info         : this function is for defining outputs of model, there names and shapes

        @return       : name maped to variable which contain inside a tensor needed shape
        @rtype        : dict(str : torch.autograd.Variable(torch.FloatTensor(), requires_grad=False))  
        '''
        inputs = self.get_inputs()
        return self.dict_forward(inputs)

