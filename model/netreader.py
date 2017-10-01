from base.reader import Reader
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, data_path, transform):
        super(MyDataset, self).__init__()
        self.aggr_dataset = MNIST(data_path, download=True, train=True, transform=transform)
    def __len__(self):
        return self.aggr_dataset.__len__()
    def __getitem__(self, idx):
        X, y = self.aggr_dataset.__getitem__(idx)
        X = X.view(-1)
        y = torch.FloatTensor(1).fill_(y)
        return {'image' : X, 'label' : y}


class NetReader(Reader):
    def __init__(self, params):
        super(NetReader, self).__init__(params)
        self.dataset = MyDataset(self.data_path, self.transform) 

    def visualize(self, x, y, loss):
        pass
