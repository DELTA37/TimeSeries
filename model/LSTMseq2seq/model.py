import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
from base.layers import *
from base.network import BaseNet
import os

class Net(BaseNet):
    '''
    @info             : this place is for global parameter (hyperparameters) definition
    '''

    def __init__(self, params, *args, **kwargs):
        super(Net, self).__init__(params, *args, **kwargs)
        '''
        @info         : this place is for model architecture assignment, and use self.params and self parameters in model definition

        @param params : global parameters model configuration from config file
        @type  params : dict()

        @return       : constructor
        @rtype        : None
        '''
        '''
        description:
        we will do seq2seq using lstm window_size -> window_size
        '''
        self.window_size    = params['window_size']

        self.conv1          = Conv1dLayer(1, 1, 3, padding = 1, bias=True)
        self.relu1          = ReLULayer()
        self.conv2          = Conv1dLayer(1, 1, 3, padding = 1, bias=True)
        self.relu2          = ReLULayer()
     
    def get_inputs(self):
        '''
        @info         : this function is for defining inputs of model, there names and shapes

        @return       : name maped to variable which contain inside a tensor needed shape
        @rtype        : dict(str : torch.autograd.Variable(torch.FloatTensor(), requires_grad=False))  
        '''
        return {'price' : Variable(torch.randn(self.params['batch_size'], 1, self.window_size), requires_grad=False)}
   

    def dict_forward(self, x):
        '''
        @info         : it is just implementation of out model, so we apply defined operations to input tensor x

        @param  x     : input tensor
        @type   x     : torch.autograd.Variable()

        @return       : output of our model after all operations
        @rtype        : dict(str : torch.autograd.Variable())
        '''
        old         = x['price']
        x           = self.conv1(x['price'])
        x           = self.relu1(x)
        x           = self.conv2(x)
        x           = self.relu2(x)
        return {'label' : x, 'x' : old}

    def get_criterion(self, params):
        '''
        @info         : this function is for loss definition

        @param params : all information from config file
        @type  params : dict

        @return       : loss function, which takes (y, y_pred, ...) and gives result of loss function on current object
        @rtype        : callable
        '''
        class MSELoss_fn(nn.Module):
            def __init__(self, params):
                super(MSELoss_fn, self).__init__()
                self.aggr_loss = nn.MSELoss() # you can provide your own loss function
            def forward(self, y_pred, y):
                return self.aggr_loss(y_pred['label'], y['label']) + self.aggr_loss(y_pred['label'], y_pred['x'])
            def backward(self):
                return self.aggr_loss.backward()
                
        return MSELoss_fn(self.params)



