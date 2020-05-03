import torch
from difflr import CONFIG
from difflr.utils import plot_information_transfer
from difflr.models import LinearClassifierDSC
from difflr.data import MNISTDataset
import numpy as np
import random

CONFIG.DRY_RUN = True

def epoch_end_hook(model:LinearClassifierDSC):
    # print('Variance: ', np.mean([param.var().item() for param in model.edge_weights]))
    edge_weights = [param.item() for param in model.edge_weights]
    plot_information_transfer(edge_weights)


def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'dsc_ffn',
        "num_classes": 10,
        'in_features': 784,
        'epochs': 10,
        'batch_size': 256,
        'lr': 1e-3,
        'dnn_config':
            {

                'layers': [100, 50, 10]
            }
    }

    model = LinearClassifierDSC(config=config)
    model.fit(dataset=MNISTDataset, epoch_end_hook=epoch_end_hook)


if __name__ == '__main__':
    main()
