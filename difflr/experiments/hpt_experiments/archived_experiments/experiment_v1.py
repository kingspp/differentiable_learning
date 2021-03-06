import torch

from difflr.models import LinearClassifier
from difflr.data import MNISTDataset
from difflr import CONFIG
from difflr.experiments import Tuner
import time

CONFIG.DRY_RUN = False

def main():
    torch.manual_seed(0)

    start_time = time.time()
    config = {
         'model_name': 'simpleffn_tuned',
         "num_classes": 10,
         'in_features': 784,
         'epochs': [10, 25, 50, 100],
         'batch_size': [10, 32, 64, 256],
         'lr': [1e-1, 1e-2, 1e-3],
         'lr_decay': False,
         "train_p": 100,
         "test_p": 100,
         'dnn_config':
             {

                 'layers': [10, 5, 10]
             }
     }

    model = LinearClassifier
    tuner = Tuner(config=config, model=model)
    tuner.tune(dataset=MNISTDataset, cv_split=5)
    print(f"Finished tuning in {time.time()-start_time}secs")

if __name__ == '__main__':
    main()