import torch.nn as nn
import torch
from base.layers import Layer
from torch.autograd import Variable

class ShapeRegister:
    D = dict()
    @staticmethod
    def registerShape(func_name, shape_in, shape_out):
        ShapeRegister.D[func_name] = (list(shape_in), list(shape_out))

def ShapeOutput(in_shape, layer):
    x = Variable(torch.randn(*in_shape))
    layer = layer.replace(' ', '')
    layer_name = layer[0:layer.find('(')]
    arg_name       = layer[layer.find('(')+1:-1]
    lst = arg_name.split(',')
    kwargs = []
    args = []
    for l in lst:
        if '=' in l:
            l = l.replace('=', ':')
            kwargs.append(l)
        else:
            args.append(l)

    kwargs = eval('{' + ','.join(kwargs) + '}')
    args =   eval('(' + ','.join(args) + ')')
    print(kwargs, args)
    f = Layer.AccessableMethods[layer_name](*args, **kwargs)
    y = f(x)
    return y.data.numpy().shape



