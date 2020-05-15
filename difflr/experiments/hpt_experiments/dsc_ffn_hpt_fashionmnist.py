import torch
from difflr.models import LinearClassifierDSC
from difflr.data import FashionMNISTDataset
from difflr import CONFIG
from difflr.experiments import Tuner
import time

CONFIG.DRY_RUN = False


def main():
    torch.manual_seed(0)

    start_time = time.time()
    config = {
        'model_name': 'fashion_mnist_dsc_tuned_adam_100p_data_100p_params',
        "num_classes": 10,
        'in_features': 784,
        'epochs': 100,
        'batch_size': [256],
        'lr': [1e-2, 1e-3],
        'lr_decay': [False, 1, 0.1],
        'reg_coeff': [False, 1, 0.1, 0.01, 0.001],
        "train_p": 100,
        "test_p": 100,
        'dnn_config':
            {

                'layers': [100, 50, 10]
            },
        'early_stopping': True,
        'patience': 5
    }

    model = LinearClassifierDSC
    tuner = Tuner(config=config, model=model)
    tuner.tune(dataset=FashionMNISTDataset, cv_split=5, test_interval=1)
    print(f"Finished tuning in {time.time() - start_time}secs")

    ###########################################################

    start_time = time.time()
    config = {
        'model_name': 'fashion_mnist_dsc_tuned_adam_100p_data_10p_params',
        "num_classes": 10,
        'in_features': 784,
        'epochs': 100,
        'batch_size': [256],
        'lr': [1e-2, 1e-3],
        'lr_decay': [False, 1, 0.1],
        'reg_coeff': [False, 1, 0.1, 0.01, 0.001],
        "train_p": 100,
        "test_p": 100,
        'dnn_config':
            {

                'layers': [10, 10, 10]
            },
        'early_stopping': True,
        'patience': 5
    }

    model = LinearClassifierDSC
    tuner = Tuner(config=config, model=model)
    tuner.tune(dataset=FashionMNISTDataset, cv_split=5, test_interval=1)
    print(f"Finished tuning in {time.time() - start_time}secs")

    ###########################################################
    start_time = time.time()
    config = {
        'model_name': 'fashion_mnist_dsc_tuned_adam_10p_data_100p_params',
        "num_classes": 10,
        'in_features': 784,
        'epochs': 100,
        'batch_size': [32],
        'lr': [1e-2, 1e-3],
        'lr_decay': [False, 1, 0.1],
        'reg_coeff': [False, 1, 0.1, 0.01, 0.001],
        "train_p": 100,
        "test_p": 100,
        'dnn_config':
            {

                'layers': [100, 50, 10]
            },
        'early_stopping': True,
        'patience': 5
    }

    model = LinearClassifierDSC
    tuner = Tuner(config=config, model=model)
    tuner.tune(dataset=FashionMNISTDataset, cv_split=5, data_per=10, test_interval=1)
    print(f"Finished tuning in {time.time() - start_time}secs")

    ###########################################################
    start_time = time.time()
    config = {
        'model_name': 'fashion_mnist_dsc_tuned_adam_10p_data_10p_params',
        "num_classes": 10,
        'in_features': 784,
        'epochs': 100,
        'batch_size': [32],
        'lr': [1e-2, 1e-3],
        'lr_decay': [False, 1, 0.1],
        'reg_coeff': [False, 1, 0.1, 0.01, 0.001],
        "train_p": 100,
        "test_p": 100,
        'dnn_config':
            {

                'layers': [10, 10, 10]
            },
        'early_stopping': True,
        'patience': 5
    }

    model = LinearClassifierDSC
    tuner = Tuner(config=config, model=model)
    tuner.tune(dataset=FashionMNISTDataset, cv_split=5, data_per=10, test_interval=1)
    print(f"Finished tuning in {time.time() - start_time}secs")


if __name__ == '__main__':
    main()
