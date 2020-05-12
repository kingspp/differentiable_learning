import torch

from difflr.models import LinearClassifier
from difflr.data import CIFARDataset


def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'cifar_simple_ffn',
        "exp_dir":"/tmp/difflr",
        "num_classes": 10,
        'in_features': 32*32,
        'epochs': 10,
        'batch_size': 32,
        'lr': 1e-2,
        "train_p":1,
        "test_p":100,
        'dnn_config':
            {

                'layers': [40, 25, 10]
            }
    }

    model = LinearClassifier(config=config)
    model.fit(dataset=CIFARDataset)


if __name__ == '__main__':
    main()
