from base.reader import Reader
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import torch
import cv2
import os
import scipy
import numpy as np
import scipy.ndimage

class MyDataset(Dataset):
    name = "MNIST for dual learning"
    def __init__(self, data_path, transform, X_dim, z_dim):
        super(MyDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform
        self.mnist = MNIST(root=data_path, download=True, train=True)
        self.z_dim = z_dim
        self.X_dim = X_dim
    def __len__(self):
        return 25000
    def __getitem__(self, idx):

        z1 = torch.randn(self.z_dim)
        z2 = torch.randn(self.z_dim)
        X1 = torch.FloatTensor(np.array(self.mnist[idx][0], dtype=np.float32).reshape(28*28))
        X2 = np.array(self.mnist[len(self) - idx - 1][0]).reshape((-1,28,28))
        X2 = scipy.ndimage.interpolation.rotate(X2, 90, axes=(1, 2))
        X2 = torch.FloatTensor(np.float32(X2.reshape(28*28)))

        return {'X1' : X1, 'X2' : X2, 'z1' : z1, 'z2' : z2}


class NetReader(Reader):
    def __init__(self, params):
        super(NetReader, self).__init__(params)
        self.dataset = MyDataset(self.data_path, self.transform, params['X_dim'], params['z_dim']) 

    def visualize(self, x, y, loss):
        pass
