import torch.nn as nn
import torch
from base.layers import Layer
import ast

class SingletonDecorator:
    def __init__(self, cls):
        self.cls = cls
        self.instance = None
    def __call__(self, *args, **kwargs):
        if self.instance == None:
            self.instance = self.cls(*args, **kwargs)
        return self.instance

@SingletonDecorator
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

    def infoLayer(self, name):
        pass


'''
Chain := variable -> Function -> ... 
Function := Layer | Function -> ...
Expression := new_variable = Chain | Chain -> new_variable | new_function = Function
valid:
new_variable = shape
decled:
input : variable 
output : variable
model : function 
'''

class Var:
    def __init__(self, name, shape=None, chain_id=-1):
        self.name = name
        self.shape = shape
        chain_id = chain_id

class Const:
    def __init__(self, name, val):
        self.name = name
        self.val = np.array(val)

class Func:
    def __init__(self, name, shape_in=None, shape_out=None, arg_names=[], body=[]):
        self.name = name
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.arg_names = arg_names
        self.body = body

class Chain:
    def __init__(self, chain_id, var_input, body=[]):
        self.chain_id = chain_id
        self.var_input = var_input
        self.body = body

class Expr:
    def __init__(self, var, chain_id):
        self.var = var
        self.chain_id = chain_id

@SingletonDecorator
class StateMachine: # fabric pattern for previous small classes

    @staticmethod
    def rawParse(expr_str):
        return list(map(lambda x : str.replace(x, ' ', ''), expr_str.split('->')))
    
    @staticmethod
    def argParse(expr_str):
        return list(map(lambda x : str.replace(x, ' ', ''), expr_str.split(',')))

    @staticmethod
    def funcLeftParse(expr_str):
        expr_str = expr_str.replace(' ', '')
        name = expr_str[:expr_str.find('(')]
        expr_str = expr_str[expr_str.find('(')+1:-1]
        arg_names = StateMachine.argParse(expr_str)
        return (name, arg_names)
    
    @staticmethod
    def funcRightParse(expr_str):
        expr_str = expr_str.replace(' ', '')
        name = expr_str[:expr_str.find('(')]
        expr_str = expr_str[expr_str.find('(')+1:-1]
        arg_names = StateMachine.argParse(expr_str)
        lst1 = []
        lst2 = dict()
        for i in range(len(args_names)):
            if args_names[i].find('=') == -1:
                lst1.append(args_names[i])
            else:
                var_name = args_names[:args_names[i].find('=')]
                var_val = args_name[args_names[i].find('=')+1:]
                lst2[var_name] = var_val
        return (name, lst1, lst2)

    def __init__(self, params):
        self.vars = dict({'input' : Var('input'), 'output' : Var('output')})
        self.consts = dict()
        self.funcs = dict('model' : Func('model'))
        self.chains = dict()
        self.exprs = dict()

    def addVariable(self, name, shape=None, chain_id=-1):
        assert(name not in self.vars.keys())
        self.vars[name] = {'shape' : shape, 'chain_id' : chain_id}

    def setShape(self, name, shape):
        if name in self.vars.keys():
            assert(self.vars[name]['shape'] == None)
            self.vars[name]['shape'] = shape
        else:
            self.addVariable(name, shape)

    def setFunction(self, name, shape_in=None, shape_out=None, decl=[]):
        assert(name not in self.funcs.keys())
        self.funcs[name] = Func(name, shape_in, shape_out, decl)

    def deduceType(self, expr_str):
        expr_str = expr_str.replace(' ', '')
        if expr_str in self.vars.keys():
            return 'Variable'
        elif expr_str in self.consts.keys():
            return 'Constant'
        else:
            # TODO
            raise NotImplemented

    def parseExpr(self, expr_str):
        lvalue = expr_str[:expr_str.find('=')]
        rvalue = expr_str[expr_str.find('=')+1:]
        #TODO
        pass
