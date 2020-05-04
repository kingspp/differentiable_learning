# -*- coding: utf-8 -*-
"""
@created on: 5/3/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""
from difflr.models import Model
import torch
from prettytable import PrettyTable


def shape_printer_hook(model: Model, input_shape):
    """ Pass the instance of Model class and input shape to print the output shapes of each layer of model """

    table = PrettyTable()
    table.field_names = ['Layer Name', 'Shape']
    table.add_row(['Input shape', input_shape])
    x = torch.rand(input_shape)
    for e, layer in enumerate(model.conv_layers):
        x = layer(x)
        table.add_row([f'Conv {e}', x.shape])
    x = x.view(-1, x.shape[1:].numel())
    table.add_row([f'Flattened', x.shape])
    for e, layer in enumerate(model.linear_layers):
        x = layer(x)
        table.add_row([f'Linear {e}', x.shape])
    print(table)
