import torch

from difflr.models import LinearClassifierGSC
from difflr.data import FashionMNISTDataset
from difflr import CONFIG

CONFIG.DRY_RUN = False

def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'fashion_mnist_gsc_ffn',
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

    model = LinearClassifierGSC(config=config)
    model.fit(dataset=FashionMNISTDataset)


if __name__ == '__main__':
    main()
