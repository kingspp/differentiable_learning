import torch

from difflr.models import LinearClassifierGSC
from difflr.data import MNISTDataset
from difflr import CONFIG

CONFIG.DRY_RUN = False


def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'gsc_ffn_10p_100p',
        "num_classes": 10,
        'in_features': 784,
        'epochs': 10,
        'batch_size': 256,
        'lr': 0.1,
        "train_p": 10,
        "test_p": 100,
        'dnn_config':
            {

                'layers': [100, 50, 10]
            }
    }

    model = LinearClassifierGSC(config=config)
    model.fit(dataset=MNISTDataset)


if __name__ == '__main__':
    main()
