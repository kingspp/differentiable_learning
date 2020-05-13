import torch

from difflr.models import LinearClassifier, LinearClassifierGSC, LinearClassifierDSC
from difflr.data import CIFARDataset
from difflr import CONFIG

CONFIG.DRY_RUN = False


def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'simpleffn_cifar_100p_data_100p_params',
        "num_classes": 10,
        'in_features': 1024,
        'epochs': 100,
        'batch_size': 32,
        'lr': 1e-2,
        'lr_decay': False,
        "train_p": 100,
        "test_p": 100,
        'dnn_config':
            {

                'layers': [150, 100, 10]
            },
        'early_stopping': True,
        'patience': 5
    }

    model = LinearClassifier(config=config)
    model.fit(dataset=CIFARDataset, test_interval=1)

    config = {
        'model_name': 'gsc_cifar_100p_data_100p_params',
        "num_classes": 10,
        'in_features': 1024,
        'epochs': 100,
        'batch_size': 32,
        'lr': 1e-2,
        'lr_decay': False,
        "train_p": 100,
        "test_p": 100,
        'dnn_config':
            {

                'layers': [150, 100, 10]
            },
        'early_stopping': True,
        'patience': 5
    }

    model = LinearClassifierGSC(config=config)
    model.fit(dataset=CIFARDataset, test_interval=1)

    config = {
        'model_name': 'dsc_cifar_100p_data_100p_params',
        "num_classes": 10,
        'in_features': 1024,
        'epochs': 100,
        'batch_size': 32,
        'lr': 1e-2,
        'lr_decay': False,
        "train_p": 100,
        "test_p": 100,
        'dnn_config':
            {

                'layers': [150, 100, 10]
            },
        'early_stopping': True,
        'patience': 5
    }

    model = LinearClassifierDSC(config=config)
    model.fit(dataset=CIFARDataset, test_interval=1)

    config = {
        'model_name': 'dsc_cifar_100p_data_10p_params',
        "num_classes": 10,
        'in_features': 1024,
        'epochs': 100,
        'batch_size': 32,
        'lr': 1e-2,
        'lr_decay': False,
        "train_p": 100,
        "test_p": 100,
        'dnn_config':
            {

                'layers': [15, 10, 10]
            },
        'early_stopping': True,
        'patience': 5
    }

    model = LinearClassifierDSC(config=config)
    model.fit(dataset=CIFARDataset, test_interval=1)


if __name__ == '__main__':
    main()
