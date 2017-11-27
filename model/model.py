import argparse
import torch
import torch.nn as nn
from torch.optim import RMSprop
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
        self.X_dim = params['X_dim']
        self.z_dim = params['z_dim']
        self.h_dim = params['h_dim']
        self.lam1  = params['lam1']
        self.lam2  = params['lam2']

        self.G1 = SequentialBlock(
                LinearLayer(self.X_dim + self.z_dim, self.h_dim),
                ReLULayer(),
                LinearLayer(self.h_dim, self.X_dim),
                SigmoidLayer(),
        )

        self.G2 = SequentialBlock(
                LinearLayer(self.X_dim + self.z_dim, self.h_dim),
                ReLULayer(),
                LinearLayer(self.h_dim, self.X_dim),
                SigmoidLayer(),
        )

        self.D1 = SequentialBlock(
            LinearLayer(self.X_dim, self.h_dim),
            ReLULayer(),
            LinearLayer(self.h_dim, 1),
        )

        self.D2 = SequentialBlock(
            LinearLayer(self.X_dim, self.h_dim),
            ReLULayer(),
            LinearLayer(self.h_dim, 1),
        )


    def get_inputs(self):
        '''
        @info         : this function is for defining inputs of model, there names and shapes

        @return       : name maped to variable which contain inside a tensor needed shape
        @rtype        : dict(str : torch.autograd.Variable(torch.FloatTensor(), requires_grad=False))  
        '''
        return {
            'X1' : Variable(torch.randn(*([self.params['batch_size']] + [self.X_dim])), requires_grad=False), 
            'X2' : Variable(torch.randn(*([self.params['batch_size']] + [self.X_dim])), requires_grad=False), 
            'z1' : Variable(torch.randn(*([self.params['batch_size']] + [self.z_dim])), requires_grad=False), 
            'z2' : Variable(torch.randn(*([self.params['batch_size']] + [self.z_dim])), requires_grad=False), 
        }
   

    def dict_forward(self, x):
        '''
        @info         : it is just implementation of out model, so we apply defined operations to input tensor x

        @param  x     : input tensor
        @type   x     : torch.autograd.Variable()

        @return       : output of our model after all operations
        @rtype        : dict(str : torch.autograd.Variable())
        '''

        G1_X1 = self.G1(torch.cat([x['X1'], x['z1']], 1))
        D1_real = self.D1(x['X2'])
        D1_fake = self.D1(G1_X1)

        G2_X2 = self.G2(torch.cat([x['X2'], x['z2']], 1))
        D2_real = self.D2(x['X2'])
        D2_fake = self.D2(G2_X2)

        G2_G1_X1 = self.G2(G1_X1)
        G1_G2_X2 = self.G1(G2_X2)

        return {
            'X1'      : x['X1'],
            'D1_real' : D1_real,
            'D1_fake' : D1_fake,

            'X2'      : x['X2'],
            'D2_real' : D2_real,
            'D2_fake' : D2_fake,
            
            'X1_recon': G2_G1_X1,
            'X2_recon': G1_G2_X2,
        }

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
            def forward(self, y_pred, y):
                loss = -(torch.mean(y_pred['D1_real']) - torch.mean(y_pred['D1_fake']))
                loss -= (torch.mean(y_pred['D2_real']) - torch.mean(y_pred['D2_fake']))
                loss -= torch.mean(y_pred['D1_fake']) + torch.mean(y_pred['D2_fake'])
                loss += self.lam1 * torch.mean(torch.sum(torch.abs(y_pred['X1_recon'] - y_pred['X1']), 1))
                loss += sefl.lam2 * torch.mean(torch.sum(torch.abs(y_pred['X2_recon'] - y_pred['X2']), 1))
                return loss

        return Loss()

    def get_optim(self):
        class Opt:
            def __init__(self):
                self.closure_bool = 0 
                self.optD1 = RMSprop(self.D1.get_trainable(), self.params['lr'])
                self.optD2 = RMSprop(self.D2.get_trainable(), self.params['lr'])
                self.optG =  RMSprop(self.G1.get_trainable() + self.G2.get_trainable(), self.params['lr'])

            def zero_grad(self):
                self.optD1.zero_grad()
                self.optD2.zero_grad()
                self.optG.zero_grad()

            def step(self):
                self.optD1.step()
                self.optD2.step()
                self.optG.step()
            
            def call(self, context):
                pass
        self.opt = Opt()
        return self.opt
