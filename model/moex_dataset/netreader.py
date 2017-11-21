from base.reader import Reader
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import torch
import cv2
import os
import pandas as pd
import numpy as np


class MoexDataset(Dataset):
    name = "moex dataset"

    def __init__(self, data_path, transform, window_size):
        super(MyDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform
        self.window_size = window_size
        self.data = np.load(self.data_path)
        self.len = len(self.data)
    def __len__(self):
        return self.len - self.window_size - 1
    def __getitem__(self, idx):
        arr   = np.array(self.data[idx:idx+self.window_size])
        n     = np.array(self.data[idx+self.window_size+1])
        prob  = int(np.mean(arr) >= n)

        return {'price' : torch.FloatTensor(arr), 'label' : torch.FloatTensor([n])}

class NetReader(Reader):
    def __init__(self, params):
        super(NetReader, self).__init__(params)
        self.dataset = MyDataset(self.data_path, self.transform, params['window_size']) 

    def visualize(self, x, y, loss):
        pass
