import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
from base.layers import *
from base.network import BaseNet

class Net(BaseNet):
    '''
    @info             : this place is for global parameter (hyperparameters) definition
    '''
    IMAGE_SHAPE = [784,]
    LABEL_SHAPE = [1,]
    def __init__(self, params, *args, **kwargs):
        super(Net, self).__init__(params, *args, **kwargs)
        '''
        @info         : this place is for model architecture assignment, and use self.params and self parameters in model definition

        @param params : global parameters model configuration from config file
        @type  params : dict()

        @return       : constructor
        @rtype        : None
        '''
        self.lin  = LinearLayer(self.params['batch_size'], 1)
        self.lin1 = LinearLayer(784, 1)
        self.sig1 = SigmoidLayer()
        
    def get_inputs(self):
        '''
        @info         : this function is for defining inputs of model, there names and shapes

        @return       : name maped to variable which contain inside a tensor needed shape
        @rtype        : dict(str : torch.autograd.Variable(torch.FloatTensor(), requires_grad=False))  
        '''
        return {'image' : Variable(torch.randn(*([self.params['batch_size']] + Net.IMAGE_SHAPE)), requires_grad=False)}
   
    def get_outputs(self):
        '''
        @info         : this function is for defining outputs of model, there names and shapes

        @return       : name maped to variable which contain inside a tensor needed shape
        @rtype        : dict(str : torch.autograd.Variable(torch.FloatTensor(), requires_grad=False))  
        '''
        return {'label' : Variable(torch.randn(*([self.params['batch_size']] + Net.LABEL_SHAPE)), requires_grad=False)}

    def dict_forward(self, x):
        '''
        @info         : it is just implementation of out model, so we apply defined operations to input tensor x

        @param  x     : input tensor
        @type   x     : torch.autograd.Variable()

        @return       : output of our model after all operations
        @rtype        : dict(str : torch.autograd.Variable())
        '''
        x = self.lin1(x['image'])
        x = self.sig1(x)
        return {'label' : x}

    def get_criterion(self, params):
        '''
        @info         : this function is for loss definition

        @param params : all information from config file
        @type  params : dict

        @return       : loss function, which takes (y, y_pred, ...) and gives result of loss function on current object
        @rtype        : callable
        '''
        class Loss(nn.Module):
            def __init__(self):
                super(Loss, self).__init__()
                self.aggr_loss = torch.nn.MSELoss() # you can provide your own loss function
            def forward(self, y_pred, y):
                return self.aggr_loss.forward(y_pred['label'], y['label']) # for a dict values passed to __call__
            def backward(self):
                return self.aggr_loss.backward()
        return Loss()



