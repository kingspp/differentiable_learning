# -*- coding: utf-8 -*-
"""
@created on: 5/3/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import torch

from difflr.models import LinearClassifierGSC
from difflr.data import FashionMNISTDataset


def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'gsc_ffn',
        "num_classes": 10,
        'in_features': 784,
        'epochs': 50,
        'batch_size': 256,
        'lr': 1e-3,
        'dnn_config':
            {

                'layers': [100, 50, 10]
            }
    }

    model = LinearClassifierGSC(config=config)
    model.fit(dataset=FashionMNISTDataset)


if __name__ == '__main__':
    main()
