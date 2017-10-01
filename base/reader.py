from PIL import Image
import cv2
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class Reader:
    def __init__(self, params):
        '''
        Initialize reader with config-file-parameters
        '''
        if   params['kind'] == 'train':
            self.data_path = params['train_path']
        elif params['kind'] == 'test':
            self.data_path = params['test_path']
        else:
            raise NotImplemented

        transform_dict = params['transforms']
        lst = []
        for key, val in transform_dict.items(): # create a transforms from config
            if   key == "ToTensor":
                lst.append(transforms.ToTensor())
            elif key == "Normalize":
                lst.append(transforms.Normalize(val['mean'], val['std']))
            else:
                raise NotImplemented

        self.transform          = transforms.Compose(lst)
        self.dataset            = None # you need override this member
        self.batch_size         = params['batch_size']
        self.shuffle            = params['shuffle']
        self.result_test_dir    = params['result_test_dir']

    def getDataLoader(self):
        '''
        return data loader of transfored data
        '''
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    def ImageVisualization(self, image, y, loss):
        '''
        Embedded way to visualize your data
        '''
        pass
        
