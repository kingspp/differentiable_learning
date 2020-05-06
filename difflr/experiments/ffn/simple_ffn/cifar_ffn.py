import torch

from difflr.models import LinearClassifier
from difflr.data import CIFARDataset


def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'cifar_simple_ffn',
        "num_classes": 10,
        'in_features': 32*32,
        'epochs': 10,
        'batch_size': 256,
        'lr': 1e-2,
        "train_p":100,
        "test_p":100,
        'dnn_config':
            {

                'layers': [100, 50, 10]
            }
    }

    model = LinearClassifier(config=config)
    model.fit(dataset=CIFARDataset)


if __name__ == '__main__':
    main()
