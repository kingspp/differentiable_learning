import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from difflr.models import Model
import numpy as np
from difflr.data import MNISTDataset
from difflr.utils import plot_information_transfer



class Binarize(torch.autograd.Function):
    """Custom rounding of PyTorch tensors
    Differentiable binarization with straight-through gradient estimation.
    """

    @staticmethod
    def forward(ctx, x):
        # print(x.round())
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        # print(sum(grad_output))
        return grad_output


class ConnectionWeightBasedPruning(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        if 'IndexBackward' in t.grad_fn.__str__():
            return default_mask
        else:
            return self.current_mask.reshape(default_mask.shape)



def prune(module, name, mask):
    """Prunes tensor corresponding to parameter called `name` in `module`
    by removing every other entry in the tensors.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    `name+'_orig'`.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module

    Examples:
    """
    ConnectionWeightBasedPruning.current_mask = mask
    ConnectionWeightBasedPruning.apply(module, name)
    return module


class LinearClassifierDSCPruned(Model):
    def __init__(self, config):
        super().__init__(name='dsc_ffn_pruned', config=config)
        self.layers = nn.ModuleList([])
        self.relu_activation = torch.nn.ReLU()
        self.softmax_activation = torch.nn.LogSoftmax(dim=-1)
        self.concat_nodes = 0
        self.edge_weights = []
        self.binarize = Binarize.apply
        self.input_mask = None
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
        if self.input_mask is not None:
            mask = torch.tensor(self.input_mask, dtype=torch.float32).repeat([x.shape[0], 1])
            x = x * mask
            condition = ~(x == 0.)
            col_cond = condition.all(0)
            x = x[:, col_cond]

        inps.append(x)
        # print(f"Input: {x.shape}")
        # print(f"Weights: {self.layers[0].weight.shape}")
        # print(f"Bias: {self.layers[0].bias.shape}")

        x = self.relu_activation(self.layers[0](x * torch.sigmoid(self.edge_weights[0])))
        inps.append(x)
        for e, layer in enumerate(self.layers[1:]):
            # print(f"Input: {x.shape}")
            # print(f"Weights: {layer.weight.shape}")
            # print(f"Bias: {layer.bias.shape}")
            x = inps[0]
            for i in inps[1:]:
                x = torch.cat((i, x), 1)
            if e + 1 > len(self.layers) - 2:
                return layer(x * torch.sigmoid(self.edge_weights[e + 1])), self.softmax_activation(
                    layer(x * torch.sigmoid(self.edge_weights[e + 1])))
            else:
                x = self.relu_activation(layer(x * torch.sigmoid(self.edge_weights[e + 1])))
                inps.append(x)

    def prune(self):
        for e, (l, s) in enumerate(zip(self.layers, self.edge_weights)):
            bs = self.binarize(s).repeat([l.weight.shape[0],1])#.detach().numpy()
            prune(l, name='weight', mask=bs)

    #     print(l.weight.shape, l.bias.shape, s.shape)
    #     continue
    #     bs = self.binarize(s).detach().numpy()
    #     # w = l.weight.detach().numpy()
    #     # print(self)
    #     # print(l.weight.shape)
    #     # rmn = [1, 2, 3, 4, 5]
    #     print('lw', l.weight.shape)
    #     exit()
    #     l.weight = torch.nn.Parameter(data=torch.tensor(np.delete(bs, rmn).reshape([1, -1]), dtype=torch.float32),
    #                                   requires_grad=True)
    #     self.edge_weights[e] = torch.nn.Parameter(data=torch.tensor(np.delete(bs, rmn).reshape([1, -1]), dtype=torch.float32),
    #                                   requires_grad=True)
    #     if e == 0:
    #         mask = np.ones(784)
    #         mask[rmn] = 0
    #         self.input_mask = mask
    #     else:
    #         mask = np.ones(len(self.input_mask))
    #         print(mask.shape)
    #         mask[rmn] = 0
    #         self.input_mask = mask
    #
    #     # print(l.weight.shape)
    #     # # print(l.weight, s.shape)
    #     # print(s[s==0])
    #     # print(self.layers[0].weight.shape)
    # exit()

    def evaluate(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        prediction_probabilities = self.forward(data.reshape([-1, self.timesteps]))
        predicted_value, predicted_class = torch.max(prediction_probabilities, 1)
        return predicted_value.detach().numpy(), predicted_class.detach().numpy(), prediction_probabilities.detach().numpy()

def epoch_end_hook(model:LinearClassifierDSCPruned):
    model.prune()
    edge_weights = [torch.sigmoid(param[1]).detach().numpy() for param in model.named_parameters() if 'edge-weights-' in param[0]]
    plot_information_transfer(model, edge_weights)


if __name__ == '__main__':
    torch.manual_seed(0)

    config = {
        'model_name': 'dsc_ffn_10p',
        "num_classes": 10,
        'in_features': 784,
        'epochs': 10,
        'batch_size': 256,
        'lr': 1,
        "lr_decay":1,
        "train_p": 10,
        "test_p": 100,

        'dnn_config':
            {

                'layers': [100, 50, 10]
            }
    }

    model = LinearClassifierDSCPruned(config=config)
    # epoch_end_hook(model)
    model.fit(dataset=MNISTDataset, epoch_end_hook=epoch_end_hook)

    # print("Before: ")
    # print(model.layers[1].weight.shape)
    # layer = prune(model.layers[1], name='weight', mask=np.random.random(model.layers[1].weight.shape))
    # layer = prune(model.layers[1], name='weight', mask=np.random.random(model.layers[1].weight.shape))
    # layer = prune(model.layers[1], name='weight', mask=np.random.random(model.layers[1].weight.shape))
    # print("After: ")
    # print(layer.weight.shape)
    # print(model.layers[1].weight_mask)

    # print(model.fc3.bias_mask)