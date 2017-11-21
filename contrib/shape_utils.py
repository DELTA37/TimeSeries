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
    def _unpack(x, res):
        if isinstance(x, tuple) or isinstance(x, list):
            res0 = []
            for x0 in x:
                _unpack(x0, res0)
            res.append(res0)
        else:
            res.append(x.data.numpy().shape)
    
    def _create(shapes):
        res = []
        for i in range(len(shapes)):
            shape = shapes[i]
            if not isinstance(shape, int):
                tens = _create(list(shape))
                res.append(tens)
        if len(res) == 0:
            print(shapes)
            return Variable(torch.randn(*list(shapes)))
        return res

    x = _create(in_shape)
    layer = layer.replace(' ', '')
    layer_name = layer[0:layer.find('(')]
    arg_name       = layer[layer.find('(')+1:-1]
    lst = arg_name.split(',')
    kwargs = []
    args = []
    for l in lst:
        if '=' in l:
            kwargs.append(l)
        else:
            args.append(l)

    kwargs = eval('dict(' + ','.join(kwargs) + ')')
    args =   eval('(' + ','.join(args) + ')')
    print(kwargs, args)
    f = Layer.AccessableMethods[layer_name](*args, **kwargs)
    y = f(x)
    shapes = []
    _unpack(y, shapes)
    return shapes[0]



