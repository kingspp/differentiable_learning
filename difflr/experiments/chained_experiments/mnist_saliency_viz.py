import torch
from difflr import CONFIG
from difflr.utils import plot_information_transfer
from difflr.models import LinearClassifierDSC
from difflr.data import MNISTDataset, CIFARDataset, FashionMNISTDataset
from difflr.utils.plot_utils import visualize_input_saliency
import os
import numpy as np

CONFIG.DRY_RUN = False


def epoch_end_hook(model: LinearClassifierDSC):
    edge_weights = [torch.sigmoid(param[1]).cpu().detach().numpy() for param in model.named_parameters() if
                    'edge-weights-' in param[0]]
    _, overall_transfer = plot_information_transfer(model, edge_weights)
    if 'iv' not in model.metrics:
        model.metrics['iv'] = [overall_transfer]
    else:
        model.metrics['iv'].append(overall_transfer)

    os.system(f'mkdir -p {model.exp_dir}/viz/')
    if model.epoch_step in [1, 10, 20, 50, 100]:
        for e, edge_weight in enumerate(edge_weights):
            np.save(model.exp_dir + f'/viz/layer{e}_{model.epoch_step}', edge_weight[:model.config['in_features']])


def cleanup_hook(model: LinearClassifierDSC):
    visualize_input_saliency(data_dir=model.exp_dir + '/viz/', dataset_name=f'MNIST',
                             save_path=model.exp_dir + '/saliency.png')


def main():
    torch.manual_seed(0)
    configs = [
        # {
        #     'model_name': 'mnist_dsc_100pd_100pp_1e1_decay',
        #     "num_classes": 10,
        #     'in_features': 784,
        #     'epochs': 100,
        #     'batch_size': 32,
        #     'lr': 1e-1,
        #     'lr_decay': 1,
        #     "train_p": 100,
        #     "test_p": 100,
        #     'dnn_config':
        #         {
        #
        #             'layers': [150, 60, 10]
        #         },
        #     'early_stopping': True,
        #     'patience': 5
        # },
        #
        # {
        #     'model_name': 'mnist_dsc_100pd_100pp_1e1_decay_reg1',
        #     "num_classes": 10,
        #     'in_features': 784,
        #     'epochs': 100,
        #     'batch_size': 32,
        #     'lr': 1e-1,
        #     'lr_decay': 1,
        #     'reg_coeff': 1,
        #     "train_p": 100,
        #     "test_p": 100,
        #     'dnn_config':
        #         {
        #
        #             'layers': [150, 60, 10]
        #         },
        #     'early_stopping': True,
        #     'patience': 5
        # },
        #
        # {
        #     'model_name': 'mnist_dsc_100pd_100pp_1e1_decay_reg01',
        #     "num_classes": 10,
        #     'in_features': 784,
        #     'epochs': 100,
        #     'batch_size': 32,
        #     'lr': 1e-1,
        #     'lr_decay': 1,
        #     'reg_coeff': 0.1,
        #     "train_p": 100,
        #     "test_p": 100,
        #     'dnn_config':
        #         {
        #
        #             'layers': [150, 60, 10]
        #         },
        #     'early_stopping': True,
        #     'patience': 5
        # },
        #
        # {
        #     'model_name': 'mnist_dsc_100pd_100pp_1e1_decay_reg001',
        #     "num_classes": 10,
        #     'in_features': 784,
        #     'epochs': 100,
        #     'batch_size': 32,
        #     'lr': 1e-1,
        #     'lr_decay': 1,
        #     'reg_coeff': 0.01,
        #     "train_p": 100,
        #     "test_p": 100,
        #     'dnn_config':
        #         {
        #
        #             'layers': [150, 60, 10]
        #         },
        #     'early_stopping': True,
        #     'patience': 5
        # },
        #
        # {
        #     'model_name': 'mnist_dsc_100pd_100pp_1e2',
        #     "num_classes": 10,
        #     'in_features': 784,
        #     'epochs': 100,
        #     'batch_size': 32,
        #     'lr': 1e-2,
        #     # 'lr_decay': 1,
        #     "train_p": 100,
        #     "test_p": 100,
        #     'dnn_config':
        #         {
        #
        #             'layers': [150, 60, 10]
        #         },
        #     'early_stopping': True,
        #     'patience': 5
        # },
        #
        {
            'model_name': 'mnist_dsc_100pd_100pp_1e2_reg1',
            "num_classes": 10,
            'in_features': 784,
            'epochs': 100,
            'batch_size': 32,
            'lr': 1e-2,
            # 'lr_decay': 1,
            'reg_coeff': 1,
            "train_p": 100,
            "test_p": 100,
            'dnn_config':
                {

                    'layers': [150, 60, 10]
                },
            'early_stopping': True,
            'patience': 5
        },

        {
            'model_name': 'mnist_dsc_100pd_100pp_1e2_reg01',
            "num_classes": 10,
            'in_features': 784,
            'epochs': 100,
            'batch_size': 32,
            'lr': 1e-2,
            # 'lr_decay': 1,
            'reg_coeff': 0.1,
            "train_p": 100,
            "test_p": 100,
            'dnn_config':
                {

                    'layers': [150, 60, 10]
                },
            'early_stopping': True,
            'patience': 5
        },
        {
            'model_name': 'mnist_dsc_100pd_100pp_1e2_reg001',
            "num_classes": 10,
            'in_features': 784,
            'epochs': 100,
            'batch_size': 32,
            'lr': 1e-2,
            # 'lr_decay': 1,
            'reg_coeff': 0.01,
            "train_p": 100,
            "test_p": 100,
            'dnn_config':
                {

                    'layers': [150, 60, 10]
                },
            'early_stopping': True,
            'patience': 5
        },
        {
            'model_name': 'mnist_dsc_100pd_100pp_1e2_reg0001',
            "num_classes": 10,
            'in_features': 784,
            'epochs': 100,
            'batch_size': 32,
            'lr': 1e-2,
            # 'lr_decay': 1,
            'reg_coeff': 0.001,
            "train_p": 100,
            "test_p": 100,
            'dnn_config':
                {

                    'layers': [150, 60, 10]
                },
            'early_stopping': True,
            'patience': 5
        }
    ]

    for e, config in enumerate(configs):
        model = LinearClassifierDSC(config=config)
        epoch_end_hook(model)
        model.fit(dataset=MNISTDataset, epoch_end_hook=epoch_end_hook, cleanup_hook=cleanup_hook, test_interval=-1)
        print(f'PERCENTAGE COMPLETED: {(e + 1) / len(configs):.2f}')


if __name__ == '__main__':
    main()
