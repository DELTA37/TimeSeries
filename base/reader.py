from PIL import Image
import cv2
import os
import numpy as np

'''
Training set is ([1,2,3, ...], [2,3,4 ...])
We provide an opportunity to transform one series to another

params is a dict
'''

class Reader:
    def __init__(self, params):
        self.c = params['series_count']
        self.n = params['in_length']
        self.m = params['out_length']
        self.initial_dataset = params['dataset']
        
        # without u - labelled dataset
        # with u - unlabelled dataset
        self.dataset = None
        self.u_dataset = None

        self.cur_file = 0
        self.u_cur_file = 0

        self.cur_obj = 0
        self.u_cur_obj = 0

        self.storage = None
        self.u_storage = None
   
    def formalize(self):
        '''
        1) Create a temp-files for a parsed dataset
        2) Set self.dataset to a list of names of created temporary files
            single temp file has the following shape (-1, c, n+m) or (-1, c, n)
            shape[0] - amount of examples
            shape[1] - amount of time series
            shape[2] - object + label for labelled
            shape[2] - object for unlabelled
        '''
        pass

    def get_object(self):
        return self.get_batch(1)

    def get_batch(self, b):
        if self.storage == None:
            self.storage = np.fromfile(self.dataset[self.cur_file]).reshape((-1, self.c, self.n + self.m))

        if self.storage.shape[0] <= self.cur_obj + b - 1:
            self.cur_file += 1
            self.cur_file %= len(self.dataset)
            self.cur_obj = 0
            self.storage = np.fromfile(self.dataset[self.cur_file]).reshape((-1, self.c, self.n + self.m))
        self.cur_obj += b
        return self.storage[self.cur_obj - b:self.cur_obj, :, 0:self.n], self.storage[self.cur_obj - b:self.cur_obj, :, self.n:]

    def get_u_object(self):
        return get_u_batch(1)

    def get_u_batch(self, b):
        if self.u_storage == None:
            self.u_storage = np.fromfile(self.u_dataset[self.u_cur_file]).reshape((-1, self.c, self.n))

        if self.u_storage.shape[0] <= self.u_cur_obj + b - 1:
            self.u_cur_file += 1
            self.u_cur_file %= len(self.u_dataset)
            self.u_cur_obj = 0
            self.u_storage = np.fromfile(self.u_dataset[self.u_cur_file]).reshape((-1, self.c, self.n))
        self.u_cur_obj += b
        return self.u_storage[self.u_cur_obj - b:self.u_cur_obj, :, :]

