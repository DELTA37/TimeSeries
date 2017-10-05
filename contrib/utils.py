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
        @type  shape_in : list()

        @param shape_out : shapes of output tensors
        @type  shape_out  : list()
        '''
        self.name = name
        self.arg_names = arg_names
        self.body = body
        self.shape_out = shape_out
        self.shapes = shapes
        self.shape_in = shape_in

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
        expr_str = expr_str.replace(' ', '')
        return list(expr_str.split('->'))
    
    @staticmethod
    def argParse(expr_str):
        expr_str = expr_str.replace(' ', '')
        if expr_str.find('(') > 0:
            expr_str = expr_str[expr_str.find('('):expr_str.find(')')]
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
        # TODO
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
        if name in self.vars.keys() or name in self.consts.keys() or name in self.funcs.keys():
            raise SyntaxError
        self.vars[name] = Var(name, shape, chain_id)

    def setShape(self, name, shape):
        if name in self.vars.keys():
            if self.vars[name]['shape'] != None:
                raise SyntaxError
            self.vars[name]['shape'] = shape
        else:
            self.setVariable(name, shape)

    def setFunction(self, name, shape_in=None, shape_out=None, args_names=[], body=[], shapes=dict()):
        if name in self.vars.keys() or name in self.consts.keys() or name in self.funcs.keys():
            raise SyntaxError
        self.funcs[name] = Func(name, shape_in, shape_out, arg_names, body, shapes)
    
    def setChain(self, var_input, body=[]):
        chain_id = len(self.chains)
        self.chains[chain_id] = Chain(chain_id, var_input, body)
        return chain_id

    def setExpr(self, var_name, chain_id):
        if self.vars[var_name].chain_id != -1:
            raise SyntaxError
        self.vars[var_name].chain_id = chain_id
        self.exprs.append(Expr(var_name, chain_id))
    
    def setConst(self, name, val):
        if name in self.vars.keys() or name in self.consts.keys() or name in self.funcs.keys():
            raise SyntaxError
        self.consts[name] = Const(name, val)
# is_correct family
    def is_correctDependies(self, body, arg_list):
        '''
        @return : True if all vars and functions in body is initialized at previous steps, elsewhere False
        @rtype  : bool
        '''
        for st in body:
            st = st.replace(' ', '')
            if StateMachine.str2tuple(st):
                continue
            if st.find('(') == -1:
                return False
            stname = st[:st.find('(')]
            stargs = st[st.find('('):]
            if stname not in self.funcs.keys() and stname not in Layer.AccessableMethods.keys():
                return False
            stparsed = StateMachine.argParse(stargs)
            for arg in stparsed:
                if str.isdigit(arg):
                    continue
                if not (arg in self.consts.keys() or arg in arg_list):
                    return False
        return True

    def is_correctChainBody(self, body):
        if not StateMachine.is_correctDependies(body, []):
            return False
        if body[0] not in self.vars.keys():
            return False
        return True
    
    def is_correctLeftFunctionArgs(self, arg_list):
        for i in range(len(arg_list)):
            arg_list[i] = arg_list[i].replace(' ', '')
            if arg_list[i] in self.vars.keys() or str.isdigit(arg_list[i][0]):
                return False
        return True

# deduce family
    def deduceLeftType(self, lvalue, rvalue):
        rtype = deduceRightType(rvalue)
        if rtype[0] == 'Const':
            return ['ConstantAssignment', lvalue, rtype]
        if rtype[0] == 'Shape':
            return ['VariableShapeAssignment', lvalue, rtype]
        elif rtype[0] == 'Chain':
            return ['VariableChainAssighment', lvalue, rtype]
        if lvalue[:lvalue.find('(')] != -1 or rtype[0] == 'Function':
            return ['FunctionAssignment', lvalue[:lvalue.find('(')], lvalue[lvalue.find('('):], rtype]
        raise SyntaxError

    def deduceRightType(self, rvalue):
        rvalue = rvalue.replace(' ', '')
        tpl = StateMachine.str2tuple(rvalue)
        if tpl:
            return ('Shape', tpl)
        
        if str.isdigit(rvalue):
            return ('Const', int(rvalue))

        body = StateMachine.rawParse(rvalue)
        tpl = StateMachine.str2tuple(body[0])
        if tpl:
            return ('Function', body)
        if body[0] in self.vars.keys():
            return ('Chain', body)
        if body[0][0:body[0].find('(')] in self.vars.keys():
            return ('Function', body)
        raise SyntaxError

    def parseExpr(self, expr_str):
        expr_str = expr_str.replace(' ', '')
        lvalue = expr_str[:expr_str.find('=')]
        rvalue = expr_str[expr_str.find('=')+1:]
        types = deduceLeftType(lvalue, rvalue)
        
        if types[0] == 'VariableShapeAssignment':
            lvalue      = types[1]
            shp_str     = types[2][1]
            shape       = str2tuple(shp_str)
            if shape == False:
                raise SyntaxError
            else:
                StateMachine.setShape(lvalue, shape)
        elif types[0] == 'VariableChainAssighment':
            lvalue      = types[1]
            body        = types[2][1]
            if not StateMachine.is_correctChainBody(body):
                raise SyntaxError
            chain_id = StateMachine.setChain(body[0], body)
            StateMachine.setVariable(lvalue, chain_id)
            StateMachine.setExpr(lvalue, chain_id)

        elif types[0] == 'FunctionAssignment':
            fname       = types[1]
            fargs       = StateMachine.argParse(types[2])
            body        = types[3][1]
            if not StateMachine.is_correctDependies(body, fargs):
                raise SyntaxError
            if not StateMachine.is_correctLeftFunctionArgs(fargs):
                raise SyntaxError
            tpl = StateMachine.str2tuple(body[0])
            if tpl:
                StateMachine.setFunction(fname, tpl, None, fargs, body, None)
            else:
                StateMachine.setFunction(fname, None, None, fargs, body, None)
        elif types[0] == 'ConstantAssignment':
            name = types[1]
            val  = types[2]
            StateMachine.setConst(name, val)

