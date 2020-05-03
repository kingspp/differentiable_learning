import torch
from difflr import CONFIG
from difflr.utils import plot_information_transfer
from difflr.models import LinearClassifierDSC
from difflr.data import MNISTDataset
import numpy as np
import random

CONFIG.DRY_RUN = True

def epoch_end_hook(model:LinearClassifierDSC):
    edge_weights = [torch.sigmoid(param[1]).detach().numpy() for param in model.named_parameters() if 'edge-weights-' in param[0]]
    plot_information_transfer(model, edge_weights)



def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'dsc_ffn_50',
        "num_classes": 10,
        'in_features': 784,
        'epochs': 10,
        'batch_size': 256,
        'lr': 1e-2,
        'dnn_config':
            {

                'layers': [100, 50, 10]
            }
    }

    model = LinearClassifierDSC(config=config)
    epoch_end_hook(model)
    model.fit(dataset=MNISTDataset, epoch_end_hook=epoch_end_hook)


if __name__ == '__main__':
    main()
