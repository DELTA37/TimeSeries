import sys, warnings
import inspect
from termcolor import colored
import time

class DeprecateWrapper:
    def __init__(self, x, deprecations=[]):
        self.obj = x
        self.depr = deprecations
    
    def __getitem__(self, key):
        caller = inspect.stack()[1][3]
        if key in self.depr and caller == '__init__':
            print(colored("ERROR:", 'red'))
            warnings.warn("Property %s is deprecated in model declaration" % key)
            time.sleep(10)
        return self.obj[key]
    
    def __getattr__(self, key):
        caller = inspect.stack()[1][3]
        if attr in self.depr and caller == '__init__':
            print(colored("ERROR:", 'red'))
            warnings.warn("Property %s is deprecated in model declaration" % key)
            time.sleep(10)
        return getattr(self.obj, key)





