import torch

from difflr.models import LinearClassifier
from difflr.data import FashionMNISTDataset


def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'fashion_mnist_simple_ffn',
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

    model = LinearClassifier(config=config)
    model.fit(dataset=FashionMNISTDataset)


if __name__ == '__main__':
    main()
