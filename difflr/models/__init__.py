from torch import nn
import torch
from abc import ABCMeta, abstractmethod
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from difflr.utils import generate_timestamp, Tee
import os
from difflr import DIFFLR_EXPERIMENTS_RUNS_PATH
from itertools import count
import json
from difflr.utils import CustomJsonEncoder
import time
from torchsummary import summary
from difflr import CONFIG
from collections import deque


class Model(nn.Module, metaclass=ABCMeta):
    def __init__(self, name: str, config):
        super().__init__()
        self.name = name
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.id = config['model_name'] + '-' + generate_timestamp()
        if 'exp_dir' not in self.config:
            self.exp_dir = os.path.join(DIFFLR_EXPERIMENTS_RUNS_PATH, self.id)
        else:
            self.exp_dir = os.path.join(self.config['exp_dir'], self.id)
        if not CONFIG.DRY_RUN:
            os.system(f'mkdir -p "{self.exp_dir}"')
        self.metrics = {
            'time_elapsed': '',
            "timestamp": self.id.split('-')[-1],
            "config": self.config,
            'train': {
                'batch': {
                    'loss': [],
                    'accuracy': [],
                    'mse': []
                },
                'epoch': {
                    'loss': [],
                    'accuracy': [],
                    'mse': []
                }
            },
            "valid": {
                'epoch': {
                    'loss': [],
                    'accuracy': [],
                    'mse': []
                }
            }
            , 'test': {
                'loss': '',
                'accuracy': '',
                'mse': []
            }
        }
        if 'train_p' not in self.config:
            self.config['train_p'] = 90
            self.config['valid_p'] = 10
        elif 'valid_p' not in self.config:
            self.config['valid_p'] = int(self.config['train_p'] * 0.1)
            self.config['train_p'] = self.config['train_p'] - self.config['valid_p']

        if 'test_p' not in self.config:
            self.config['test_p'] = 100

        if 'lr_decay' not in self.config:
            self.config['lr_decay'] = False

    @abstractmethod
    def evaluate(self, data):
        pass

    def run_train(self, train_loader, valid_loader, log_type, log_interval, batch_end_hook, epoch_end_hook,
                  shape_printer_hook, early_stopper=None):
        self.train()

        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        # self.writer.add_graph(self.to(self.device), images.to(self.device))

        print('Model: \n')
        print(self)
        print("\n")

        print('Summary: \n')
        summary(self, input_size=images.shape[1:])
        print('\n')
        self.optimizer = optim.Adam(self.parameters(), lr=self.config['lr'])
        # self.optimizer = optim.SGD(self.parameters(), lr=self.config['lr'])

        if self.config["lr_decay"] is not False:
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer,
                                                                       gamma=self.config['lr_decay'])

        global_step = count()
        start_time = time.time()
        for epoch in tqdm.tqdm(range(1, self.config['epochs'] + 1)):
            train_metrics = {'loss': [], 'acc': [], 'mse': []}
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                if epoch == 1 and batch_idx == 0 and shape_printer_hook is not None:
                    self.shape_printer_hook(data[1:].shape)
                self.optimizer.zero_grad()
                logits = self(data)
                loss = F.nll_loss(logits, target, reduction='mean')

                # l2_reg = None
                # for weight in self.edge_weights:
                #     if l2_reg is None:
                #         l2_reg = torch.sigmoid(weight).norm(2)
                #     else:
                #         l2_reg=l2_reg+ torch.sigmoid(weight).norm(2)
                # loss= loss+ l2_reg * 0.1
                train_metrics['loss'].append(loss.item())
                loss.backward(retain_graph=True)
                self.optimizer.step()
                train_metrics['acc'].append(
                        accuracy_score(y_true=target.cpu(), y_pred=torch.max(logits, axis=1).indices.cpu()))
                # train_metrics['mse'].append(
                        # mse_score(logits=torch.softmax(raw, dim=1), target=target,
                        #           num_classes=self.config['num_classes'],
                        #           reduction='mean', device=self.device).item())
                if not CONFIG.DRY_RUN:
                    self.writer.add_scalar(tag='Train/batch/loss', scalar_value=train_metrics['loss'][-1],
                                           global_step=next(global_step))
                    self.writer.add_scalar(tag='Train/batch/accuracy', scalar_value=train_metrics['acc'][-1],
                                           global_step=next(global_step))
                    # self.writer.add_scalar(tag='Train/batch/mse', scalar_value=train_metrics['mse'][-1],
                    #                        global_step=next(global_step))
                    self.metrics['train']['batch']['loss'].append(train_metrics['loss'][-1])
                    self.metrics['train']['batch']['accuracy'].append(train_metrics['acc'][-1])
                    # self.metrics['train']['batch']['mse'].append(train_metrics['mse'][-1])

                if (log_type == 'batch' and batch_idx % log_interval == 0):
                    print(
                            f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                            f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {train_metrics["loss"][-1]:.6f}\tAcc: '
                            f'{train_metrics["acc"][-1]:.4f}\t'
                            # f'MSE: {train_metrics["mse"][-1]:.4f}'
                    )
                batch_end_hook(self)
            train_metrics_mean = {k: np.mean(v) for k, v in train_metrics.items()}

            if (log_type == 'epoch' and epoch % log_interval == 0):
                self.run_test(test_loader=valid_loader, mode='valid', step=epoch)
                print(
                        f'Train Epoch: {epoch} | Loss: {train_metrics_mean["loss"]:.6f} '
                        f'| Acc: {train_metrics_mean["acc"]:.4f}'
                        # f'| MSE: {train_metrics_mean["mse"]:.4f}'
                        f' || V Loss: {self.metrics["valid"]["epoch"]["loss"][-1]:.4f}'
                        f'| V Acc: {self.metrics["valid"]["epoch"]["accuracy"][-1]:.4f}'
                        # f'| V MSE: {self.metrics["valid"]["epoch"]["mse"][-1]:.4f}'
                )
            if not CONFIG.DRY_RUN:
                self.writer.add_scalar(tag='Train/epoch/loss', scalar_value=train_metrics_mean["loss"],
                                       global_step=epoch)
                self.writer.add_scalar(tag='Train/epoch/accuracy', scalar_value=train_metrics_mean["acc"],
                                       global_step=epoch)
                # self.writer.add_scalar(tag='Train/epoch/mse', scalar_value=train_metrics_mean["mse"], global_step=epoch)
                self.metrics['train']['epoch']['loss'].append(train_metrics_mean["loss"])
                self.metrics['train']['epoch']['accuracy'].append(train_metrics_mean["acc"])
                # self.metrics['train']['epoch']['mse'].append(train_metrics_mean["mse"])
            epoch_end_hook(self)
            if self.config["lr_decay"] is not False:
                self.lr_scheduler.step(epoch=epoch)

            # Early stopping
            if early_stopper is not None and early_stopper.step(self.metrics["valid"]["epoch"]["accuracy"][-1]):
                print(f'Early Stopping this run after {epoch}')
                self.metrics['time_elapsed'] = time.time() - start_time
                return
        self.metrics['time_elapsed'] = time.time() - start_time

    def run_test(self, test_loader, mode, step=None):
        metrics = {'loss': [], 'acc': [], 'mse': []}
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            logits = self(data)
            loss = F.nll_loss(logits, target, reduction='mean')
            metrics['loss'].append(loss.item())
            metrics['acc'].append(
                    accuracy_score(y_true=target.cpu(), y_pred=torch.max(logits, axis=1).indices.cpu()))
            # metrics['mse'].append(
            #         mse_score(logits=torch.softmax(raw, dim=1), target=target, num_classes=self.config['num_classes'],
            #                   reduction='mean', device=self.device).item())

        metrics_mean = {k: np.mean(v) for k, v in metrics.items()}
        if not CONFIG.DRY_RUN:
            self.writer.add_scalar(f'{mode}/loss', metrics_mean['loss'], global_step=step)
            self.writer.add_scalar(f'{mode}/accuracy', metrics_mean['acc'], global_step=step)
            # self.writer.add_scalar(f'{mode}/mse', metrics_mean['mse'], global_step=step)
            if mode == 'valid':
                self.metrics[mode]['epoch']['loss'].append(metrics_mean['loss'])
                self.metrics[mode]['epoch']['accuracy'].append(metrics_mean['acc'])
                # self.metrics[mode]['epoch']['mse'].append(metrics_mean['mse'])
            else:
                self.metrics[mode]['loss'] = metrics_mean['loss']
                self.metrics[mode]['accuracy'] = metrics_mean['acc']
                # self.metrics[mode]['mse'] = metrics_mean['mse']

    def clean(self):
        self.writer.close()
        self.writer = None

    def fit(self, dataset, log_type='epoch', log_interval=1,
            batch_end_hook=lambda x: x, epoch_end_hook=lambda x: x, shape_printer_hook=None, early_stopper=None):
        if not CONFIG.DRY_RUN:
            self.writer = SummaryWriter(f'{self.exp_dir}/graphs')
        self.to(self.device)
        with Tee(filename=self.exp_dir + '/model.log', io_enabled=not CONFIG.DRY_RUN):
            print('Config: \n', json.dumps(self.config, indent=2, cls=CustomJsonEncoder), '\n')

            train_loader, valid_loader, test_loader = dataset(batch_size=self.config['batch_size'],
                                                              train_p=self.config['train_p'],
                                                              test_p=self.config['test_p'],
                                                              valid_p=self.config['valid_p'],
                                                              use_cuda=True if self.device == torch.device(
                                                                      'cuda') else False
                                                              )
            # Train and Valid
            self.run_train(train_loader=train_loader, valid_loader=valid_loader, log_type=log_type,
                           log_interval=log_interval, batch_end_hook=batch_end_hook, epoch_end_hook=epoch_end_hook,
                           shape_printer_hook=shape_printer_hook, early_stopper=early_stopper)
            # Test
            self.run_test(test_loader=test_loader, mode="test")
            print(
                    f'Test | NLL Loss: {self.metrics["test"]["loss"]:.6f} | Acc: {self.metrics["test"]["accuracy"]:.4f} '
                    # f'| MSE: {self.metrics["test"]["mse"]}'
            )
            print(f"Took {self.metrics['time_elapsed']} seconds")

            self.clean()
            if not CONFIG.DRY_RUN:
                self.save()
                json.dump(self.metrics, open(self.exp_dir + '/metrics.json', 'w'), cls=CustomJsonEncoder, indent=2)

        return self.metrics

    def save(self):
        torch.save(self, f'{self.exp_dir}/{self.name}.ckpt')


class LinearClassifier(Model):
    def __init__(self, config):
        super().__init__(name='simple_ffn', config=config)
        self.layers = nn.ModuleList([])
        self.relu_activation = torch.nn.ReLU()
        self.softmax_activation = torch.nn.LogSoftmax(dim=-1)
        for e, node in enumerate(self.config['dnn_config']["layers"]):
            prev_node = config["in_features"] if e == 0 else self.config['dnn_config']["layers"][e - 1]
            self.layers.extend([nn.Linear(prev_node, node)])

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        for layer in self.layers[:-1]:
            x = self.relu_activation(layer(x))
        x = self.layers[-1](x)
        return self.softmax_activation(x)

    def evaluate(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        prediction_probabilities = self.forward(data.reshape([-1, self.timesteps]))
        predicted_value, predicted_class = torch.max(prediction_probabilities, 1)
        return predicted_value.detach().numpy(), predicted_class.detach().numpy(), prediction_probabilities.detach().numpy()


class LinearClassifierGSC(Model):
    def __init__(self, config):
        super().__init__(name='gsc_ffn', config=config)
        self.layers = nn.ModuleList([])
        self.relu_activation = torch.nn.ReLU()
        self.softmax_activation = torch.nn.Softmax(dim=-1)
        self.concat_nodes = 0
        for e, node in enumerate(self.config['dnn_config']["layers"]):
            if e == 0:
                prev_node = config["in_features"]
            else:
                prev_node += self.config['dnn_config']["layers"][e - 1]
            self.layers.extend([nn.Linear(prev_node, node)])

    def forward(self, x):
        inps = []
        x = x.reshape([x.shape[0], -1])
        inps.append(x)
        x = self.relu_activation(self.layers[0](x))
        inps.append(x)
        for e, layer in enumerate(self.layers[1:]):
            x = inps[0]
            for i in inps[1:]:
                x = torch.cat((i, x), 1)
            if e + 1 > len(self.layers) - 2:
                x = layer(x)
                return x, self.softmax_activation(x)
            else:
                x = self.relu_activation(layer(x))
                inps.append(x)

    def evaluate(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        prediction_probabilities = self.forward(data.reshape([-1, self.timesteps]))
        predicted_value, predicted_class = torch.max(prediction_probabilities, 1)
        return predicted_value.detach().numpy(), predicted_class.detach().numpy(), prediction_probabilities.detach().numpy()


class LinearClassifierDSC(Model):
    def __init__(self, config):
        super().__init__(name='dsc_ffn', config=config)
        self.layers = nn.ModuleList([])
        self.relu_activation = torch.nn.ReLU(inplace=False)
        self.softmax_activation = torch.nn.LogSoftmax(dim=-1)
        self.concat_nodes = 0
        self.edge_weights = []
        for e, node in enumerate(self.config['dnn_config']["layers"]):
            if e == 0:
                prev_node = config["in_features"]
            else:
                prev_node += self.config['dnn_config']["layers"][e - 1]
            self.edge_weights.append(
                    torch.nn.Parameter(data=torch.tensor(np.full(shape=[prev_node], fill_value=1), dtype=torch.float32),
                                       requires_grad=True))
            self.register_parameter(f'edge-weights-{e}', self.edge_weights[-1])

            self.layers.extend([nn.Linear(prev_node, node)])

    def forward(self, x):
        inps = []
        x = x.reshape([x.shape[0], -1])
        inps.append(x)
        x = self.relu_activation(self.layers[0](x * torch.sigmoid(self.edge_weights[0])))
        inps.append(x)
        for e, layer in enumerate(self.layers[1:]):
            x = inps[0]
            for i in inps[1:]:
                x = torch.cat((i, x), 1)
            if e  == len(self.layers)-2:
                x = layer(x * torch.sigmoid(self.edge_weights[e + 1]))
                return self.softmax_activation(x)
            else:
                x = self.relu_activation(layer(x * torch.sigmoid(self.edge_weights[e + 1])))
                inps.append(x)

    def evaluate(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        prediction_probabilities = self.forward(data.reshape([-1, self.timesteps]))
        predicted_value, predicted_class = torch.max(prediction_probabilities, 1)
        return predicted_value.detach().numpy(), predicted_class.detach().numpy(), prediction_probabilities.detach().numpy()


class CNNClassifier(Model):
    def __init__(self, config):
        super().__init__(name='simple_ffn', config=config)
        self.layers = nn.ModuleList([])
        self.relu_activation = torch.nn.ReLU()
        self.softmax_activation = torch.nn.Softmax(dim=-1)
        for e, node in enumerate(self.config['dnn_config']["layers"]):
            prev_node = config["in_features"] if e == 0 else self.config['dnn_config']["layers"][e - 1]
            self.layers.extend([nn.Linear(prev_node, node)])

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        for layer in self.layers[:-1]:
            x = self.relu_activation(layer(x))
        x = self.layers[-1](x)
        return x, self.softmax_activation(x)

    def evaluate(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        prediction_probabilities = self.forward(data.reshape([-1, self.timesteps]))
        predicted_value, predicted_class = torch.max(prediction_probabilities, 1)
        return predicted_value.detach().numpy(), predicted_class.detach().numpy(), prediction_probabilities.detach().numpy()
