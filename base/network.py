import tensorflow as tf
import cv2

class Network:
    def __init__(self, params):
        self.pre_transform = [] # list of callable
        self.post_transform = [] # list of callable
        self.inputs = dict() # placeholder
        self.labels = dict() # placeholder
        self.outputs = dict() # operation
        self.production = dict() # operaion

    def get_inputs(self):
        return self.inputs

    def get_labels(self):
        return self.labels

    def get_outputs(self):
        return self.outputs

    def get_production(self):
        return self.production
    
    def get_loss(self):
        pass

    def transform(self, inputs, restore):
        '''
        main model
        '''
        pass

    def __call__(self, inputs, restore):
        return self.transform(inputs, restore)
    
