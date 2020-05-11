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
    total_train_size = 60_000
    total_test_size = 10_000

    def __new__(cls, batch_size=32, use_cuda=False, train_p=90, valid_p=10, test_p=100, *args, **kwargs):
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        if isinstance(train_p, int):
            TRAIN_SAMPLES = int((60000 * train_p) / 100)
            TRAIN_P = train_p
        else:
            TRAIN_SAMPLES = len(train_p)
            TRAIN_P = len(train_p) / 60000 * 100

        if isinstance(valid_p, int):
            VALID_SAMPLES = int((60000 * valid_p) / 100)
            VALID_P = valid_p
        else:
            VALID_SAMPLES = len(valid_p)
            VALID_P = len(valid_p) / 60000 * 100

        TEST_SAMPLES = int((10000 * test_p) / 100)
        train_sampler = SubsetRandomSampler(range(0, TRAIN_SAMPLES) if isinstance(train_p, int) else train_p)
        valid_sampler = SubsetRandomSampler(
                range(TRAIN_SAMPLES, TRAIN_SAMPLES + VALID_SAMPLES) if isinstance(valid_p, int) else valid_p)
        test_sampler = SubsetRandomSampler(range(0, TEST_SAMPLES))
        print(
                f"MNIST Dataset:\nTraining on {TRAIN_SAMPLES} samples ({TRAIN_P}% of 60000) \n"
                f"Valid on {VALID_SAMPLES} samples ({VALID_P}% of 60000)\n"
                f"Test on {TEST_SAMPLES} samples ({test_p}% of 10000)\n")
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(DIFFLR_DATA_PATH, train=True, download=True, transform=transform),
                batch_size=batch_size, sampler=train_sampler, **kwargs)

        valid_loader = torch.utils.data.DataLoader(
                datasets.MNIST(DIFFLR_DATA_PATH, train=True, download=True, transform=transform),
                batch_size=batch_size, sampler=valid_sampler, **kwargs)

        test_loader = torch.utils.data.DataLoader(
                datasets.MNIST(DIFFLR_DATA_PATH, train=False, transform=transform),
                batch_size=batch_size, sampler=test_sampler, **kwargs)
        return train_loader, valid_loader, test_loader


class FashionMNISTDataset():
    total_train_size = 60_000
    total_test_size = 10_000

    def __new__(cls, batch_size, use_cuda, train_p=90, valid_p=10, test_p=100, *args, **kwargs):
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        if isinstance(train_p, int):
            TRAIN_SAMPLES = int((60000 * train_p) / 100)
            TRAIN_P = train_p
        else:
            TRAIN_SAMPLES = len(train_p)
            TRAIN_P = len(train_p) / 60000 * 100

        if isinstance(valid_p, int):
            VALID_SAMPLES = int((60000 * valid_p) / 100)
            VALID_P = valid_p
        else:
            VALID_SAMPLES = len(valid_p)
            VALID_P = len(valid_p) / 60000 * 100

        TEST_SAMPLES = int((10000 * test_p) / 100)
        train_sampler = SubsetRandomSampler(range(0, TRAIN_SAMPLES) if isinstance(train_p, int) else train_p)
        valid_sampler = SubsetRandomSampler(
                range(TRAIN_SAMPLES, TRAIN_SAMPLES + VALID_SAMPLES) if isinstance(valid_p, int) else valid_p)
        test_sampler = SubsetRandomSampler(range(0, TEST_SAMPLES))
        print(
                f"FASHIONMNIST Dataset:\nTraining on {TRAIN_SAMPLES} samples ({TRAIN_P}% of 60000) \n"
                f"Valid on {VALID_SAMPLES} samples ({VALID_P}% of 60000)\n"
                f"Test on {TEST_SAMPLES} samples ({test_p}% of 10000)\n")
        train_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST(DIFFLR_DATA_PATH, train=True, download=True, transform=transform),
                batch_size=batch_size, sampler=train_sampler, **kwargs)

        valid_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST(DIFFLR_DATA_PATH, train=True, download=True, transform=transform),
                batch_size=batch_size, sampler=valid_sampler, **kwargs)

        test_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST(DIFFLR_DATA_PATH, train=False, transform=transform),
                batch_size=batch_size, sampler=test_sampler, **kwargs)
        return train_loader, valid_loader, test_loader


class CIFARDataset():
    total_train_size = 50_000
    total_test_size = 10_000

    def __new__(cls, batch_size, use_cuda, train_p=90, valid_p=10, test_p=100, *args, **kwargs):
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        transform = transforms.Compose(
                [transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
                 transforms.Normalize((0.5,), (0.5,))])
        if isinstance(train_p, int):
            TRAIN_SAMPLES = int((60000 * train_p) / 100)
            TRAIN_P = train_p
        else:
            TRAIN_SAMPLES = len(train_p)
            TRAIN_P = len(train_p) / 60000 * 100

        if isinstance(valid_p, int):
            VALID_SAMPLES = int((60000 * valid_p) / 100)
            VALID_P = valid_p
        else:
            VALID_SAMPLES = len(valid_p)
            VALID_P = len(valid_p) / 60000 * 100

        TEST_SAMPLES = int((10000 * test_p) / 100)
        train_sampler = SubsetRandomSampler(range(0, TRAIN_SAMPLES) if isinstance(train_p, int) else train_p)
        valid_sampler = SubsetRandomSampler(
                range(TRAIN_SAMPLES, TRAIN_SAMPLES + VALID_SAMPLES) if isinstance(valid_p, int) else valid_p)
        test_sampler = SubsetRandomSampler(range(0, TEST_SAMPLES))
        print(
                f"CIFAR-10 Dataset:\nTraining on {TRAIN_SAMPLES} samples ({TRAIN_P}% of 60000) \n"
                f"Valid on {VALID_SAMPLES} samples ({VALID_P}% of 60000)\n"
                f"Test on {TEST_SAMPLES} samples ({test_p}% of 10000)\n")
        train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(DIFFLR_DATA_PATH, train=True, download=True, transform=transform),
                batch_size=batch_size, sampler=train_sampler, **kwargs)

        valid_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(DIFFLR_DATA_PATH, train=True, download=True, transform=transform),
                batch_size=batch_size, sampler=valid_sampler, **kwargs)

        test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(DIFFLR_DATA_PATH, train=False, transform=transform),
                batch_size=batch_size, sampler=test_sampler, **kwargs)
        return train_loader, valid_loader, test_loader
