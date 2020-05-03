import torch

from difflr.models import LinearClassifierGSC
from difflr.data import MNISTDataset


def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'gsc_ffn',
        'in_features': 784,
        'epochs': 10,
        'batch_size': 256,
        'lr': 1e-3,
        'dnn_config':
            {

                'layers': [100, 50, 10]
            }
    }

    model = LinearClassifierGSC(config=config)
    print(model)
    # model.fit(dataset=MNISTDataset)


if __name__ == '__main__':
    main()
