# -*- coding: utf-8 -*-
"""
@created on: 5/5/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

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
from difflr import CONFIG
from difflr.models.cnn_models import DSCCNN
from difflr.data import FashionMNISTDataset

CONFIG.DRY_RUN = False


def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'dsc_cnn_fashionmnist_10p_params_5ep_10p_data',
        "num_classes": 10,
        'in_features': [1, 28, 28],
        'epochs': 5,
        'batch_size': 128,
        'lr': 1e-2,
        'dnn_config':
            {
                'kernel_size': [3, 3, 3, 3],
                'stride': [2, 1, 1, 1],
                'filters_maps': [3, 6, 12, 6],
                'linear': [300, 10]
            },
        'train_p': 10,
        'test_p': 100,
    }

    model = DSCCNN(config=config)
    model.fit(dataset=FashionMNISTDataset, shape_printer_hook=model.shape_printer_hook)


if __name__ == '__main__':
    main()
