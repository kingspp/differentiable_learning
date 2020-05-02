from torch import nn
import torch
from abc import ABCMeta, abstractmethod
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from sklearn.metrics import accuracy_score
import numpy as np


class Model(nn.Module, metaclass=ABCMeta):
    def __init__(self, config):
        super().__init__()
        self.name = config["model_name"]
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def evaluate(self, data):
        pass

    def fit(self, epochs, batch_size, data, log_type='epoch', log_interval=1, lr=0.001):
        self.train()
        train_loader, test_loader = data(batch_size=batch_size, use_cuda=True if self.device == 'cuda' else False)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
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
                if (log_type == 'batch' and batch_idx % log_interval == 0):
                    print(
                        f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                        f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tAcc: {acc}')
            if (log_type == 'epoch' and epoch % log_interval == 0):
                print(
                    f'Train Epoch: {epoch} | Loss: {np.mean(train_loss_batch):.6f} | Acc: {np.mean(train_acc_batch):.4f}')

        test_loss_batch, test_acc_batch = [], []
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self(data)
            loss = F.nll_loss(output, target)
            test_loss_batch.append(loss.item())
            loss.backward()            
            acc = accuracy_score(y_true=target, y_pred=torch.max(output, axis=1).indices)
            test_acc_batch.append(acc)
        print(
            f'Test | Loss: {np.mean(test_loss_batch):.6f} | Acc: {np.mean(test_acc_batch):.4f}')

    def save(self):
        torch.save(self, f'{self.name}.ckpt')

1
class LinearClassifier(Model):
    def __init__(self, config):
        super().__init__(config=config)
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
