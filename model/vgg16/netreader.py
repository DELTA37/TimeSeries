from base.reader import Reader
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import torch
import cv2
import os

class MyDataset(Dataset):
    name = "Cat Dog Dataset"
    url  = "https://www.kaggle.com/c/dogs-vs-cats"
    def __init__(self, data_path, transform):
        super(MyDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform
    def __len__(self):
        return 25000
    def __getitem__(self, idx):
        prefix = 'cat.'
        y = 0
        if (idx >= 12500):
            idx -= 12500
            prefix = 'dog.'
            y = 1
        fname = os.path.expanduser(os.path.join(self.data_path, prefix + str(idx) + '.jpg'))
        print(fname)
        X = cv2.imread(fname, cv2.IMREAD_COLOR) 
        X = cv2.resize(X, (3 * 32, 3 * 32))
        X = self.transform(X)
        return {'image' : X, 'label' : y}


class NetReader(Reader):
    def __init__(self, params):
        super(NetReader, self).__init__(params)
        self.dataset = MyDataset(self.data_path, self.transform) 

    def visualize(self, x, y, loss):
        pass
