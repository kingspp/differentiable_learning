import json
import random
import numpy as np
import torch
from difflr import DIFFLR_DATA_PATH
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler


class Dataset(torch.utils.data.Dataset):
    def __init__(self, name="Dataset"):
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.valid_data = None
        self.valid_label = None
        self.train_meta = None
        self.test_meta = None
        self.valid_meta = None
        self.indices = {}
        self.name = name

    def _create_valid_from_train(self, valid_ratio: 0.2):
        val_index = self.train_data.shape[0] - int(self.train_data.shape[0] * valid_ratio)
        self.train_data, self.train_label = self.train_data[:val_index], self.train_label[:val_index]
        self.valid_data, self.valid_label = self.train_data[val_index:], self.train_label[val_index:]

    def __getitem__(self, ix):
        return self.train_data[ix], self.train_label[ix]

    def __len__(self):
        return len(self.train_data)

    def get_random_sample(self, cls=None):
        # todo Validation
        if cls is None:
            r = random.randrange(0, len(self.train_data))
            return self.train_data[r], self.train_label[r], r
        else:
            if cls not in self.indices:
                self.indices[cls] = np.where(self.train_label == cls)[0]
            r = random.randrange(0, len(self.indices[cls]))
            return self.train_data[self.indices[cls][r]], self.train_label[self.indices[cls][r]], self.indices[cls][
                r]

    def read_meta(self, file_list):
        meta_data = []
        for file in file_list:
            meta_data.append(json.load(open(file)))
        return meta_data

    def batch(self, batch_size=32):
        """
        Function to batch the data
        :param batch_size: batches
        :return: batches of X and Y
        """
        l = len(self.train_data)
        for ndx in range(0, l, batch_size):
            yield self.train_data[ndx:min(ndx + batch_size, l)], self.train_label[ndx:min(ndx + batch_size, l)]


class MNISTDataset():
    def __new__(cls, batch_size, use_cuda, *args, **kwargs):
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(DIFFLR_DATA_PATH, train=True, download=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(DIFFLR_DATA_PATH, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        return train_loader, test_loader


class FashionMNISTDataset():
    def __new__(cls, batch_size, use_cuda, *args, **kwargs):
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(DIFFLR_DATA_PATH, train=True, download=True,
                                  transform=transforms.Compose(
                                      [transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,))])),
            batch_size=batch_size, shuffle=False, **kwargs)  # , sampler=SubsetRandomSampler(range(0, 100))
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(DIFFLR_DATA_PATH, train=False, transform=transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5,), (0.5,))])),
            batch_size=batch_size, shuffle=False, **kwargs)  # , sampler=SubsetRandomSampler(range(0, 100))
        return train_loader, test_loader


class CIFARDataset():
    def __new__(cls, batch_size, use_cuda, *args, **kwargs):
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(DIFFLR_DATA_PATH, train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=False, **kwargs)  # , sampler=SubsetRandomSampler(range(0, 100))
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(DIFFLR_DATA_PATH, train=False, transform=transform),
                                                  batch_size=batch_size, shuffle=False,
                                                  **kwargs)  # , sampler=SubsetRandomSampler(range(0, 100))
        return train_loader, test_loader
