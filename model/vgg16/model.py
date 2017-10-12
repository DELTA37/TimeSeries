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
    IMAGE_SHAPE = [3, 3 * 32, 3 * 32]
    LABEL_SHAPE = [1]
    def __init__(self, params, *args, **kwargs):
        super(Net, self).__init__(params, *args, **kwargs)
        '''
        @info         : this place is for model architecture assignment, and use self.params and self parameters in model definition

        @param params : global parameters model configuration from config file
        @type  params : dict()

        @return       : constructor
        @rtype        : None
        '''
        self.layer11 = Conv2d_BN_ReLULayer(3, 64, 3)
        self.layer12 = Conv2d_BN_ReLULayer(64, 64, 3)
        self.pool1  = MaxPool2dLayer(2)
        self.layer21 = Conv2d_BN_ReLULayer(64, 128, 3)
        self.layer22 = Conv2d_BN_ReLULayer(128, 128, 3)
        self.layer23 = Conv2d_BN_ReLULayer(128, 128, 3)
        self.pool2  = MaxPool2dLayer(2)
        self.layer31 = Conv2d_BN_ReLULayer(128, 256, 3)
        self.layer32 = Conv2d_BN_ReLULayer(256, 256, 3)
        self.layer33 = Conv2d_BN_ReLULayer(256, 256, 3)
        self.pool3  = MaxPool2dLayer(2)
        self.layer41 = Conv2d_BN_ReLULayer(256, 512, 3)
        self.layer42 = Conv2d_BN_ReLULayer(512, 512, 3)
        self.layer43 = Conv2d_BN_ReLULayer(512, 512, 3)
        self.pool4  = MaxPool2dLayer(2)
        self.fclayer1 = Conv2dLayer(512, 9 * 512, 3, stride=2)
        self.fclayer2 = Conv2dLayer(9 * 512, 9 * 512, 1)
        self.fclayer3 = Conv2dLayer(9 * 512, 1, 1)
        self.sig = SigmoidLayer()
     
    def get_inputs(self):
        '''
        @info         : this function is for defining inputs of model, there names and shapes

        @return       : name maped to variable which contain inside a tensor needed shape
        @rtype        : dict(str : torch.autograd.Variable(torch.FloatTensor(), requires_grad=False))  
        '''
        return {'image' : Variable(torch.randn(*([self.params['batch_size']] + Net.IMAGE_SHAPE)), requires_grad=False)}
   

    def dict_forward(self, x):
        '''
        @info         : it is just implementation of out model, so we apply defined operations to input tensor x

        @param  x     : input tensor
        @type   x     : torch.autograd.Variable()

        @return       : output of our model after all operations
        @rtype        : dict(str : torch.autograd.Variable())
        '''
        x = self.layer11(x['image'])
        x = self.layer12(x)
        x = self.pool1(x)
        print("1")
        x = self.layer21(x)
        x = self.layer22(x)
        x = self.pool2(x)
        print("2")
        x = self.layer31(x)
        x = self.layer32(x)
        x = self.pool3(x)
        print("3")
        x = self.layer41(x)
        x = self.layer42(x)
        x = self.pool4(x)
        print("4")
        x = self.fclayer1(x)
        x = self.fclayer2(x)
        x = self.fclayer3(x)
        x = self.sig(x)
        x = x.view(-1)
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
                self.aggr_loss = torch.nn.NLLLoss() # you can provide your own loss function
            def forward(self, y_pred, y):
                return self.aggr_loss.forward(y_pred['label'], y['label']) # for a dict values passed to __call__
            def backward(self):
                return self.aggr_loss.backward()
        return Loss()



