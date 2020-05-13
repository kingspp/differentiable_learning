import torch
from difflr import CONFIG
from difflr.utils import plot_information_transfer
from difflr.models import LinearClassifierDSC
from difflr.data import MNISTDataset, CIFARDataset, FashionMNISTDataset

CONFIG.DRY_RUN = False

def epoch_end_hook(model:LinearClassifierDSC):
    edge_weights = [torch.sigmoid(param[1]).cpu().detach().numpy() for param in model.named_parameters() if 'edge-weights-' in param[0]]
    _, overall_transfer = plot_information_transfer(model, edge_weights)
    if 'iv' not in model.metrics:
        model.metrics['iv'] = [overall_transfer]
    else:
        model.metrics['iv'].append(overall_transfer)



def main():
    torch.manual_seed(0)

    config = {
        'model_name': 'mnist_dsc_ffn_10p_data_100p_params_1e-3',
        "num_classes": 10,
        'in_features': 784,
        'epochs': 100,
        'batch_size': 32,
        'lr': 1e-3,
        # 'lr_decay': 1,
        "train_p":10,
        "test_p":100,
        'dnn_config':
            {

                'layers': [100, 50, 10]
            },
        'early_stopping': True,
        'patience': 5
    }

    model = LinearClassifierDSC(config=config)
    epoch_end_hook(model)
    model.fit(dataset=MNISTDataset, epoch_end_hook=epoch_end_hook)

    config = {
        'model_name': 'fashion_mnist_dsc_ffn_10p_data_100p_params_1e-3',
        "num_classes": 10,
        'in_features': 784,
        'epochs': 100,
        'batch_size': 32,
        'lr': 1e-3,
        # 'lr_decay': 1,
        "train_p": 10,
        "test_p": 100,
        'dnn_config':
            {

                'layers': [100, 50, 10]
            },
        'early_stopping': True,
        'patience': 5
    }

    model = LinearClassifierDSC(config=config)
    epoch_end_hook(model)
    model.fit(dataset=FashionMNISTDataset, epoch_end_hook=epoch_end_hook)

    config = {
        'model_name': 'cifar_dsc_ffn_10p_data_100p_params_1e-3',
        "num_classes": 10,
        'in_features': 1024,
        'epochs': 100,
        'batch_size': 32,
        'lr': 1e-3,
        # 'lr_decay': 1,
        "train_p": 10,
        "test_p": 100,
        'dnn_config':
            {

                'layers': [100, 50, 10]
            },
        'early_stopping': True,
        'patience': 5
    }

    model = LinearClassifierDSC(config=config)
    epoch_end_hook(model)
    model.fit(dataset=CIFARDataset, epoch_end_hook=epoch_end_hook)


if __name__ == '__main__':
    main()
