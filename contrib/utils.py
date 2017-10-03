import torch.nn as nn
import torch
from base.layers import Layer
import ast
import numpy as np

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
        '''
        @param  name        : name of variable
        @type   name        : str
        
        @param  shape       : shape of variable 
        @type   shape       : tuple, list

        @param  chain_id    : chain which associate with this variable
        @type   chain_id    : int
        '''
        self.name = str(name)
        self.shape = list(shape)
        chain_id = int(chain_id)

class Const:
    def __init__(self, name, val):
        self.name = str(name)
        self.val = np.array(val)

class Func:
    def __init__(self, name, shape_in=None, shape_out=None, arg_names=[], body=[], shapes=dict()):
        '''
        @param name : name of function
        @type  name : str

        @param shape_in : shape, which must be substitude into function, None in component if we don't know, len(shape_in) == len(arg_names)
        @type  shape_in : list(list())

        @param shape_out : shapes of output tensors
        @type  shape_out  : list()
        '''
        self.name = name
        self.arg_names = arg_names
        self.body = body
        self.shape_out = shape_out
        self.shapes = shapes
        if shape_in == None:
            self.shape_in = [[]] * len(arg_names)
        else:
            self.shape_in = shape_in

        assert(len(self.shape_in) == len(arg_names))


class Chain:
    def __init__(self, chain_id, var_input, body=[], shapes=dict()):
        self.chain_id = chain_id
        self.var_input = var_input
        self.body = body
        self.shapes = shapes

class Expr:
    def __init__(self, var_name, chain_id):
        self.var_name = var_name
        self.chain_id = chain_id


@SingletonDecorator
class StateMachine: # fabric pattern for previous small classes
# string to object functions
    @staticmethod
    def str2tuple(s):
        s = s.replace(' ', '')
        if s[0] != '(' and s[-1] != ')':
            return False
        s = s[1:-1]
        s = s.split(',')
        try:
            return list(map(lambda x : int(x), s))
        except:
            return False
         
# parse family
    @staticmethod
    def rawParse(expr_str):
        return list(map(lambda x : str.replace(x, ' ', ''), expr_str.split('->')))
    
    @staticmethod
    def argParse(expr_str):
        expr_str = expr_str.replace(' ', '')
        return list(expr_str.split(','))

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
    
    @staticmethod
    def funcBodyParse(expr_str):
        raw = StateMachine.rawParse(expr_str)
        for i in range(len(raw)):
            node = StateMachine.funcRightParse(raw[i])
            if node[0] == '':
                if i == 0:
                    shape_in = list(eval(node[1]))
                elif i == len(raw) - 1:
                    shape_out = list(eval(node[1]))
                else:
                    raise NotImplemented



# constructor
    def __init__(self, params):
        '''
        @param self.vars    : all created variables
        @type  self.vars    : dict(str : Var)

        @param self.consts  : all created constants
        @type  self.consts  : dict(str : Const)

        @param self.funcs   : all created functions
        @type  self.funcs   : dict(str : Func)

        '''
        self.vars = dict({'input' : Var('input'), 'output' : Var('output')})
        self.consts = dict()
        self.funcs = dict('model' : Func('model'))
        self.chains = dict()
        self.exprs = []

# set family
    def setVariable(self, name, shape=None, chain_id=-1):
        assert(name not in self.vars.keys())
        self.vars[name] = Var(name, shape, chain_id)

    def setShape(self, name, shape):
        if name in self.vars.keys():
            assert(self.vars[name]['shape'] == None)
            self.vars[name]['shape'] = shape
        else:
            self.addVariable(name, shape)

    def setFunction(self, name, shape_in=None, shape_out=None, decl=[]):
        assert(name not in self.funcs.keys())
        self.funcs[name] = Func(name, shape_in, shape_out, decl)
    
    def setChain(self, var_input, body=[]):
        chain_id = len(self.chains)
        self.chains[chain_id] = Chain(chain_id, var_input, body)

    def setExpr(self, var_name, chain_id):
        assert(self.vars[var_name].chain_id == -1)
        self.vars[var_name].chain_id = chain_id
        # TODO deduce shape
        self.exprs.append(Expr(var_name, chain_id))

# is_correct family
    def is_correctDependies(self, cur_var, cur_chain_id):
        '''
        @return : True if all vars and functions is initialized at previous expres, elsewhere False
        @rtype  : bool
        '''
        #TODO
        pass

    def is_correctFuncRight(self, name, args, kwargs):
        '''
        name    : str
        args    : list
        kwargs  : dict
        '''
        if name in self.funcs.keys():
            lst = args + list(kwargs.keys())
            #TODO
            self.funcs[name].args
            return True
        elif name in Layer.AccessableMethods.keys():
            return True
        else:
            return False

# deduce family
    def deduceLeftType(self, lvalue, rvalue):
        rtype = deduceRightType(rvalue)
        if rtype == 'Shape':
            return ['VariableShapeAssignment', lvalue]
        elif rtype == 'Chain':
            return ['VariableChainAssighment', lvalue]

        if lvalue[:lvalue.find('(')] != -1:
            return ['FunctionAssignment', lvalue[:lvalue.find('(')], lvalue[lvalue.find('('):]]
        raise SyntaxError

    def deduceRightType(self, rvalue):
        pass
        
    def parseExpr(self, expr_str):
        expr_str = expr_str.replace(' ', '')
        lvalue = expr_str[:expr_str.find('=')]
        rvalue = expr_str[expr_str.find('=')+1:]

        #TODO
        pass
