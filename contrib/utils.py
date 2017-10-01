import torch.nn as nn
import torch
from base.layers import Layer
import ast

class LayerFabric:
    def __init__(self, params):
        pass
    def createLayer(self, name):
        class Store:
            def __init__(self, *args, **kwargs):
                self.lst = args
                self.dct = kwargs
        name = name.replace(" ", "")
        layer_name = name[0:name.find('(')]
        args_name = name[name.find('('):]
        st = eval("Store"+args_name)
        return Layer.AccessableMethods[layer_name](*st.lst, **st.dct)
        
