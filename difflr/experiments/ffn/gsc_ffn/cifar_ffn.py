import torch

from difflr.models import LinearClassifierGSC
from difflr.data import CIFARDataset


def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'cifar_dsc',
        "num_classes": 10,
        'in_features': 32*32,
        'epochs': 10,
        'batch_size': 32,
        'lr': 1e-2,
        'lr_decay':1,
        "train_p":10,
        "test_p":100,
        'dnn_config':
            {

                'layers': [10, 10, 10]
            }
    }

    model = LinearClassifierGSC(config=config)
    model.fit(dataset=CIFARDataset)


if __name__ == '__main__':
    main()
