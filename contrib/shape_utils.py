import torch.nn as nn
import torch
from base.layers import Layer
from torch.autograd import Variable

def ShapeOutput(in_shape, layer_name, *args, **kwargs):
    x = Variable(torch.randn(*in_shape))
    f = Layer.AccessableMethods[layer_name](*args, **kwargs)
    y = f(x)
    return y.data.numpy().shape

