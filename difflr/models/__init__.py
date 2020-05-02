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
from difflr import DIFFLR_EXPERIMENTS_PATH
from itertools import count
import json
from difflr.utils import CustomJsonEncoder

class Model(nn.Module, metaclass=ABCMeta):
    def __init__(self, name: str, config):
        super().__init__()
        self.name = config["model_name"] or name
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.id = self.name + '-' + generate_timestamp()
        self.exp_dir = os.path.join(DIFFLR_EXPERIMENTS_PATH, self.name, 'runs', self.id)
        os.mkdir(path=self.exp_dir)
        self.metrics = {
            "timestamp": self.id.split('-')[-1],
            'train': {
                'batch': {
                    'loss': [],
                    'accuracy': []
                },
                'epoch': {
                    'loss': [],
                    'accuracy': [],
                }
            }, 'test': {
                'loss': '',
                'accuracy': '',
            }
        }

    @abstractmethod
    def evaluate(self, data):
        pass

    def fit(self, epochs, batch_size, data, log_type='epoch', log_interval=1, lr=0.001):
        with Tee(filename=self.exp_dir + '/model.log'):
            writer = SummaryWriter(f'{self.exp_dir}/graphs')
            self.train()

            train_loader, test_loader = data(batch_size=batch_size, use_cuda=True if self.device == 'cuda' else False)
            self.optimizer = optim.Adam(self.parameters(), lr=lr)

            global_step = count()

            for epoch in tqdm.tqdm(range(1, epochs + 1)):
                train_loss_batch, train_acc_batch = [], []
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self(data)
                    loss = F.nll_loss(output, target)
                    train_loss_batch.append(loss.item())
                    loss.backward()
                    self.optimizer.step()
                    acc = accuracy_score(y_true=target, y_pred=torch.max(output, axis=1).indices)
                    train_acc_batch.append(acc)
                    writer.add_scalar(tag='Train/batch/loss', scalar_value=loss.item(), global_step=next(global_step))
                    writer.add_scalar(tag='Train/batch/accuracy', scalar_value=acc, global_step=next(global_step))
                    self.metrics['train']['batch']['loss'].append(loss.item())
                    self.metrics['train']['batch']['accuracy'].append(acc)

                    if (log_type == 'batch' and batch_idx % log_interval == 0):
                        print(
                            f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                            f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tAcc: {acc}')

                mean_epoch_loss = np.mean(train_loss_batch)
                mean_epoch_acc = np.mean(train_acc_batch)
                if (log_type == 'epoch' and epoch % log_interval == 0):
                    print(
                        f'Train Epoch: {epoch} | Loss: {mean_epoch_loss:.6f} | Acc: {mean_epoch_acc:.4f}')
                writer.add_scalar(tag='Train/epoch/loss', scalar_value=mean_epoch_acc, global_step=epoch)
                writer.add_scalar(tag='Train/epoch/accuracy', scalar_value=mean_epoch_acc, global_step=epoch)
                self.metrics['train']['epoch']['loss'].append(mean_epoch_loss)
                self.metrics['train']['epoch']['accuracy'].append(mean_epoch_acc)

            test_loss_batch, test_acc_batch = [], []
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                loss = F.nll_loss(output, target)
                test_loss_batch.append(loss.item())
                loss.backward()
                acc = accuracy_score(y_true=target, y_pred=torch.max(output, axis=1).indices)
                test_acc_batch.append(acc)

            mean_test_loss, mean_test_acc = np.mean(train_loss_batch), np.mean(train_acc_batch)
            writer.add_scalar('Test/epoch/loss', mean_test_loss)
            writer.add_scalar('Test/epoch/accuracy', mean_test_acc)
            self.metrics['test']['loss'] = mean_test_loss
            self.metrics['test']['accuracy'] = mean_test_acc
            print(
                f'Test | Loss: {mean_test_loss:.6f} | Acc: {mean_test_acc:.4f}')

            dataiter = iter(train_loader)
            images, labels = dataiter.next()

            self.save()
            writer.add_graph(self, images)
            writer.close()
            json.dump(self.metrics, open(self.exp_dir+'/metrics.json', 'w'), cls=CustomJsonEncoder, indent=2)

    def save(self):
        torch.save(self, f'{self.exp_dir}/{self.name}.ckpt')


class LinearClassifier(Model):
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
        return self.softmax_activation(self.layers[-1](x))

    def evaluate(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        prediction_probabilities = self.forward(data.reshape([-1, self.timesteps]))
        predicted_value, predicted_class = torch.max(prediction_probabilities, 1)
        return predicted_value.detach().numpy(), predicted_class.detach().numpy(), prediction_probabilities.detach().numpy()


class CNNClassifier(Model):
    pass
