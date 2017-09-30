from PIL import Image
import cv2
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class Reader:
    def __init__(self, params):
        self.data_path = params['data_path']
        transform_dict = params['transforms']
        lst = []
        for key, val in transform_dict.items(): # create a transforms from config
            if   key == "ToTensor":
                lst.append(transforms.ToTensor())
            elif key == "Normalize":
                lst.append(transforms.Normalize(val['mean'], val['std']))
            else:
                raise NotImplemented

        self.transform  = transforms.Compose(lst)
        self.dataset    = None # you need override this member
        self.batch_size = params['batch_size']
        self.shuffle    = params['shuffle']

    def getDataLoader(self):
        '''
        return data loader of transfored data
        '''
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
