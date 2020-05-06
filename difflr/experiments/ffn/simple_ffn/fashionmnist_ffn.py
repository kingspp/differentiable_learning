import torch
from difflr import CONFIG
from difflr.models import LinearClassifier
from difflr.data import FashionMNISTDataset

CONFIG.DRY_RUN = False

def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'fashion_mnist_simple_ffn',
        "num_classes": 10,
        'in_features': 784,
        'epochs': 10,
        'batch_size': 256,
        'lr': 1e-1,
        "train_p": 10,
        "test_p": 100,
        'dnn_config':
            {

                'layers': [100, 50, 10]
            }
    }

    model = LinearClassifier(config=config)
    model.fit(dataset=FashionMNISTDataset)


if __name__ == '__main__':
    main()
