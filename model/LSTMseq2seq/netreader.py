from base.reader import Reader
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import torch
import cv2
import os
import pandas as pd
import numpy as np


class MyDataset(Dataset):
    name = "Apple Inc. stocks"
    url  = "https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=1104534000&period2=1491775200&interval=1d&events=history&crumb=TlCZBkE9JKw"
    def __init__(self, data_path, transform, window_size):
        super(MyDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform
        self.window_size = window_size
        self.data = pd.read_csv(self.data_path, sep=',')[::-1]
        self.close_price = self.data.ix[:, 'Adj Close'].tolist()
        self.len = len(self.close_price)
    def __len__(self):
        return self.len - 2*self.window_size
    def __getitem__(self, idx):
        arr   = np.array(self.close_price[idx:idx+self.window_size]).reshape((1, self.window_size))
        n     = np.array(self.close_price[idx+self.window_size:idx+2*self.window_size]).reshape((1, self.window_size))

        return {'price' : torch.FloatTensor(arr), 'label' : torch.FloatTensor(n)}

class NetReader(Reader):
    def __init__(self, params):
        super(NetReader, self).__init__(params)
        self.dataset = MyDataset(self.data_path, self.transform, params['window_size']) 

    def visualize(self, x, y, loss):
        pass
