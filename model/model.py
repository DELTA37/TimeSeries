import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
from base.layers import *
from base.network import BaseNet

class Net(BaseNet):
    IMAGE_SHAPE = [784,]
    LABEL_SHAPE = [1,]
    BATCH       = [10,]
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__(*args, **kwargs)
        '''
        description of your model
        '''
        self.lin1 = LinearLayer(784, 1)

    def get_inputs(self):
        return {'image' : Variable(torch.randn(*(Net.BATCH + Net.IMAGE_SHAPE)), requires_grad=False)}
   
    def get_outputs(self):
        return {'label' : Variable(torch.randn(*(Net.BATCH + Net.LABEL_SHAPE)), requires_grad=False)}

    def dict_forward(self, x):
        x = self.lin1(x['image'])
        return {'label' : x}

    def get_criterion(self, params):
        class Loss(nn.Module):
            def __init__(self):
                super(Loss, self).__init__()
                self.aggr_loss = torch.nn.MSELoss() # you can provide your own loss function
            def forward(self, y_pred, y):
                return self.aggr_loss.forward(y_pred['label'], y['label']) # for a dict values passed to __call__
            def backward(self):
                return self.aggr_loss.backward()
        return Loss()



