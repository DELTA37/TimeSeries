import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
from base.layers import *
from base.network import BaseNet

class Net(BaseNet):
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__(*args, **kwargs)
        '''
        description of your model
        '''
        self.conv1 = Conv1dLayer(1,1,3, restore=True, trainable=True)
        
    def forward(self, x):
        x = self.conv1(x)
        return x
    def backward(self, grad):
        grad = self.conv1.backward(grad)
        return grad


