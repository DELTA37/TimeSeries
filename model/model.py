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

        self.lstm       = nativeLSTMLayer(input_size=self.window_size, hidden_size=1, num_layers=10, bias=True)
        self.reshape1    = BatchReshapeLayer((1, self.window_size, ))
        self.conv       = Conv1dLayer(1, 1, 3, bias=True)
        self.reshape2    = BatchReshapeLayer((self.window_size - 2, ))
        self.layer2     = LinearLayer(self.window_size - 2, self.window_size - 2, bias=True)
        self.layer3     = LinearLayer(self.window_size - 2, self.window_size - 2, bias=True)
        self.layer_clf  = LinearLayer(self.window_size - 2, 2, bias=True)
        self.activation = SoftMaxLayer()
     
    def get_inputs(self):
        '''
        @info         : this function is for defining inputs of model, there names and shapes

        @return       : name maped to variable which contain inside a tensor needed shape
        @rtype        : dict(str : torch.autograd.Variable(torch.FloatTensor(), requires_grad=False))  
        '''
        return {'price' : Variable(torch.randn(self.params['batch_size'], self.window_size), requires_grad=False)}
   

    def dict_forward(self, x):
        '''
        @info         : it is just implementation of out model, so we apply defined operations to input tensor x

        @param  x     : input tensor
        @type   x     : torch.autograd.Variable()

        @return       : output of our model after all operations
        @rtype        : dict(str : torch.autograd.Variable())
        '''
        x, state    = self.lstm(x['price'])
        x           = self.reshape1(x)
        x           = self.conv(x)
        x           = self.reshape2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer_clf(x)
        x = self.activation(x)
        return {'label' : x}

    def get_criterion(self, params):
        '''
        @info         : this function is for loss definition

        @param params : all information from config file
        @type  params : dict

        @return       : loss function, which takes (y, y_pred, ...) and gives result of loss function on current object
        @rtype        : callable
        '''
        class NLLLoss_fn(nn.Module):
            def __init__(self, params):
                super(NLLLoss_fn, self).__init__()
                self.aggr_loss = nn.CrossEntropyLoss() # you can provide your own loss function
            def forward(self, y_pred, y):
                return self.aggr_loss(y_pred['label'], y['label'].view(-1)) # for a dict values passed to __call__
            def backward(self):
                return self.aggr_loss.backward()
                
        return NLLLoss_fn(self.params)



