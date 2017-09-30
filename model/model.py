import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
from base.layers import *
from base.network import BaseNet

class Net(BaseNet):
    IMAGE_SHAPE = [10, 784]
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__(*args, **kwargs)
        '''
        description of your model
        '''
        self.lin1 = LinearLayer(784, 1)

    def get_inputs(self):
        return {'image' : Variable(torch.randn(*Net.IMAGE_SHAPE), requires_grad=False)}
    
    def forward(self, x):
        x = self.lin1(x['image'])
        return x
    def backward(self, grad):
        grad = self.conv1.backward(grad)
        return grad

    def get_criterion(self, params):
        return torch.nn.MSELoss() 



