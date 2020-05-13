import torch

from difflr.models import LinearClassifierGSC
from difflr.data import FashionMNISTDataset
from difflr import CONFIG

CONFIG.DRY_RUN = False

def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'gscffn_fashion_mnist_10p_data_100p_params',
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

                'layers': [100, 50, 10]
            },
        'early_stopping': True,
        'patience': 5
    }

    model = LinearClassifierGSC(config=config)
    model.fit(dataset=FashionMNISTDataset, test_interval=1)

    config = {
        'model_name': 'gscffn_fashion_mnist_10p_data_10p_params',
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

                'layers': [10, 10, 10]
            },
        'early_stopping': True,
        'patience': 5
    }

    model = LinearClassifierGSC(config=config)
    model.fit(dataset=FashionMNISTDataset, test_interval=1)




if __name__ == '__main__':
    main()
