# -*- coding: utf-8 -*-
"""
@created on: 5/2/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import torch

from difflr.models import LinearClassifier
from difflr.data import FashionMNISTDataset


def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'simple_ffn',
        'in_features': 784,
        'epochs': 50,
        'batch_size': 256,
        'lr': 1e-3,
        'dnn_config':
            {

                'layers': [100, 50, 10]
            }
    }

    model = LinearClassifier(config=config)
    model.fit(dataset=FashionMNISTDataset)


if __name__ == '__main__':
    main()
