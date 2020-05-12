import torch

from difflr.models import LinearClassifier
from difflr.data import MNISTDataset
from difflr import CONFIG
from difflr.experiments import Tuner

CONFIG.DRY_RUN = False

def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'simpleffn_tuned',
        "num_classes": 10,
        'in_features': 784,
        'epochs': 10,
        'batch_size': 32,
        'lr': 0.01,
        'lr_decay': False,
        "train_p": 1,
        "test_p": 100,
        'dnn_config':
            {

                'layers': [40, 25, 10]
            }
    }

    model = LinearClassifier(config=config)
    model.fit(dataset=MNISTDataset)


if __name__ == '__main__':
    main()
