import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from difflr.models import LinearClassifierDSC
import numpy as np


class ConnectionWeightBasedPruning(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0
        return mask



def prune(module, name):
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
    ConnectionWeightBasedPruning.apply(module, name)
    return module


if __name__ == '__main__':
    torch.manual_seed(0)

    config = {
        'model_name': 'dsc_ffn_10p',
        "num_classes": 10,
        'in_features': 784,
        'epochs': 10,
        'batch_size': 256,
        'lr': 1e-2,
        'dnn_config':
            {

                'layers': [10, 10, 10]
            }
    }

    model = LinearClassifierDSC(config=config)

    print("Before: ")
    print(model.layers[1].weight.shape)
    layer = prune(model.layers[1], name='weight')
    print("After: ")
    print(layer.weight.shape)
    print(model.layers[1].weight_mask)

    # print(model.fc3.bias_mask)