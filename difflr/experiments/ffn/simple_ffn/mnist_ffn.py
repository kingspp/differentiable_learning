import torch

from difflr.models import LinearClassifier
from difflr.data import MNISTDataset
from difflr import CONFIG

CONFIG.DRY_RUN = False

def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'simple_ffn_100p_150_samples',
        "num_classes": 10,
        'in_features': 784,
        'epochs': 10,
        'batch_size': 256,
        'lr': 1e-3,
        "train_p":100,
        "test_p":100,
        'dnn_config':
            {

                'layers': [100, 50, 10]
            }
    }

    model = LinearClassifier(config=config)
    model.fit(dataset=MNISTDataset)


if __name__ == '__main__':
    main()
