import torch
from difflr import CONFIG
from difflr.utils import plot_information_transfer
from difflr.models import LinearClassifierDSC
from difflr.data import MNISTDataset
import numpy as np
import random
import os
from difflr import DIFFLR_RESULTS
from difflr.utils.plot_utils import visualize_input_saliency

CONFIG.DRY_RUN = False

def epoch_end_hook(model:LinearClassifierDSC):
    edge_weights = [torch.sigmoid(param[1]).detach().numpy() for param in model.named_parameters() if 'edge-weights-' in param[0]]
    _, overall_transfer = plot_information_transfer(model, edge_weights)
    if 'iv' not in model.metrics:
        model.metrics['iv'] = [overall_transfer]
    else:
        model.metrics['iv'].append(overall_transfer)
    os.system(f'mkdir -p {model.exp_dir}/viz/')
    if model.epoch_step in [1,10,20,50,100]:
        for e, edge_weight in enumerate(edge_weights):
            np.save(model.exp_dir+f'/viz/layer{e}_{model.epoch_step}', edge_weight[:model.config['in_features']])

def cleanup_hook(model:LinearClassifierDSC):
    visualize_input_saliency(data_dir=model.exp_dir+'/viz/', dataset_name=f'MNIST', save_path=model.exp_dir+'/saliency.png')


def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'mnist_dsc_ffn_10p_10p_Train',
        "num_classes": 10,
        'in_features': 784,
        'epochs': 10,
        'batch_size': 256,
        'lr':0.1,
        # 'lr_decay': 1,
        "train_p":10,
        "test_p":100,
        "reg_coeff":0,
        'dnn_config':
            {

                'layers': [100, 50, 10]
            },

    }

    model = LinearClassifierDSC(config=config)
    epoch_end_hook(model)
    model.fit(dataset=MNISTDataset, epoch_end_hook=epoch_end_hook,
              cleanup_hook=cleanup_hook)


if __name__ == '__main__':
    main()
