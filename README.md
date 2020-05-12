# Differentiable Learning
Repository for differentiable learning experiments

## Requirements:
Python 3.7+

## Installation

```bash
pip install difflr
```

## Sample Experiment - Dataset, Model and Training

```python
import torch
from difflr.models import LinearClassifier
from difflr.data import MNISTDataset
from difflr import CONFIG

CONFIG.DRY_RUN = False

torch.manual_seed(0)

config = {
    'model_name': 'simpleffn_mnist_10p_data_10p_params',
    "num_classes": 10,
    'in_features': 784,
    'epochs': 100,
    'batch_size': 32,
    'lr': 1e-2,
    'lr_decay': False,
    "train_p": 10,
    "test_p": 100,
    'dnn_config':
        {

            'layers': [40, 25, 10]
        },
    'early_stopping': True,
    'patience': 5
}

model = LinearClassifier(config=config)
model.fit(dataset=MNISTDataset, test_interval=1)
```

## Sample Hyperparameter Tuner
```python
import torch
from difflr.models import LinearClassifier
from difflr.data import CIFARDataset
from difflr import CONFIG
from difflr.experiments import Tuner
import time

CONFIG.DRY_RUN = False


torch.manual_seed(0)

start_time = time.time()
config = {
    'model_name': 'cifar_simpleffn_tuned_SGD_100p_params_100p_data',
    "num_classes": 10,
    'in_features': 1024,
    'epochs': 100,
    'batch_size': [32],
    'lr': [1e-2, 1e-3],
    'lr_decay': False,
    "train_p": 100,
    "test_p": 100,
    'dnn_config':
        {

            'layers': [150, 60, 10]
        },
    'early_stopping': True,
    'patience': 10

}

model = LinearClassifier
tuner = Tuner(config=config, model=model)
tuner.tune(dataset=CIFARDataset, cv_split=5)
print(f"Finished tuning in {time.time() - start_time} secs")
```


## Features

1. Supports MNIST, FashionMNIST and CIFAR-10 Datasets and percentage based split for cross validation

```python
from difflr.data import MNIST

train_data_loader, valid_data_loader, test_data_loader = MNIST(batch_size=32,
                                                              train_p=90,
                                                              valid_p=10,
                                                              test_p=100,                                                              
                                                              use_cuda=True if self.device == torch.device(
                                                                  'cuda') else False
                                                              )
```


2. Support Linear, LinearGSC, LinearDSC, LinearDSCPruned Models with easy run configurations
```python
from difflr.models import LinearClassifier

config = {
    'model_name': 'simpleffn_mnist_10p_data_10p_params',
    "num_classes": 10,
    'in_features': 784,
    'epochs': 100,
    'batch_size': 32,
    'lr': 1e-2,
    'lr_decay': False,
    "train_p": 10,
    "test_p": 100,
    'dnn_config':
        {

            'layers': [40, 25, 10]
        },
    'early_stopping': True,
    'patience': 5
}

model = LinearClassifier(config=config)
```
